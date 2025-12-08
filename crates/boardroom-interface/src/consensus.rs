//! Consensus mechanisms for multi-agent decision making

use crate::agent::AgentId;
use crate::message::{Message, MessageId, MessageType};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock, Mutex};
use tokio::time::{timeout, Duration};

/// Voting policy for consensus
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VotingPolicy {
    /// All agents must agree
    Unanimous,
    /// More than half must agree
    SimpleMajority,
    /// Two-thirds must agree
    SuperMajority(f64),
    /// Custom threshold (0.0 to 1.0)
    Threshold(f64),
    /// Weighted voting based on agent scores
    Weighted,
}

impl Default for VotingPolicy {
    fn default() -> Self {
        Self::SimpleMajority
    }
}

/// Consensus protocol types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConsensusProtocol {
    /// Simple voting
    SimpleVoting,
    /// Byzantine Fault Tolerant
    BFT,
    /// Proof of Stake based
    ProofOfStake,
    /// Custom protocol
    Custom,
}

/// Vote from an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    pub agent_id: AgentId,
    pub vote: bool,
    pub weight: f64,
    pub reason: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Consensus request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusRequest {
    pub id: MessageId,
    pub proposal: serde_json::Value,
    pub policy: VotingPolicy,
    pub timeout_ms: u64,
    pub initiator: AgentId,
    pub participants: Vec<AgentId>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Result of consensus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub request_id: MessageId,
    pub approved: bool,
    pub votes_for: usize,
    pub votes_against: usize,
    pub total_participants: usize,
    pub participation_rate: f64,
    pub votes: Vec<Vote>,
    pub completion_time_ms: u64,
}

/// Consensus state
#[derive(Debug)]
struct ConsensusState {
    request: ConsensusRequest,
    votes: DashMap<AgentId, Vote>,
    start_time: chrono::DateTime<chrono::Utc>,
    result_tx: broadcast::Sender<ConsensusResult>,
}

/// Consensus manager
pub struct ConsensusManager {
    active_consensus: Arc<DashMap<MessageId, Arc<ConsensusState>>>,
    agent_weights: Arc<RwLock<HashMap<AgentId, f64>>>,
    default_timeout_ms: u64,
}

impl ConsensusManager {
    /// Create new consensus manager
    pub fn new(default_timeout_ms: u64) -> Self {
        Self {
            active_consensus: Arc::new(DashMap::new()),
            agent_weights: Arc::new(RwLock::new(HashMap::new())),
            default_timeout_ms,
        }
    }

    /// Set agent weight for weighted voting
    pub async fn set_agent_weight(&self, agent_id: AgentId, weight: f64) {
        let mut weights = self.agent_weights.write().await;
        weights.insert(agent_id, weight);
    }

    /// Get agent weight
    pub async fn get_agent_weight(&self, agent_id: AgentId) -> f64 {
        let weights = self.agent_weights.read().await;
        weights.get(&agent_id).copied().unwrap_or(1.0)
    }

    /// Initiate consensus
    pub async fn initiate_consensus(
        &self,
        request: ConsensusRequest,
    ) -> anyhow::Result<broadcast::Receiver<ConsensusResult>> {
        let (tx, rx) = broadcast::channel(1);
        
        let state = Arc::new(ConsensusState {
            request: request.clone(),
            votes: DashMap::new(),
            start_time: chrono::Utc::now(),
            result_tx: tx,
        });
        
        self.active_consensus.insert(request.id, state.clone());
        
        // Start timeout task
        let active_consensus = self.active_consensus.clone();
        let timeout_ms = request.timeout_ms;
        let request_id = request.id;
        
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
            
            if let Some((_, state)) = active_consensus.remove(&request_id) {
                let _ = Self::finalize_consensus(state).await;
            }
        });
        
        Ok(rx)
    }

    /// Submit vote
    pub async fn submit_vote(&self, vote: Vote) -> anyhow::Result<()> {
        // Find the consensus request this vote is for
        for entry in self.active_consensus.iter() {
            let state = entry.value();
            
            // Check if this agent is a participant
            if state.request.participants.contains(&vote.agent_id) {
                // Add vote
                state.votes.insert(vote.agent_id, vote.clone());
                
                // Check if we have all votes
                if state.votes.len() == state.request.participants.len() {
                    // All votes received, finalize consensus
                    if let Some((_, state)) = self.active_consensus.remove(entry.key()) {
                        Self::finalize_consensus(state).await?;
                    }
                }
                
                return Ok(());
            }
        }
        
        anyhow::bail!("No active consensus found for agent {}", vote.agent_id)
    }

    /// Finalize consensus and calculate result
    async fn finalize_consensus(state: Arc<ConsensusState>) -> anyhow::Result<ConsensusResult> {
        let votes: Vec<Vote> = state.votes.iter().map(|kv| kv.value().clone()).collect();
        
        let (approved, votes_for, votes_against) = match state.request.policy {
            VotingPolicy::Unanimous => {
                let votes_for = votes.iter().filter(|v| v.vote).count();
                let approved = votes_for == state.request.participants.len();
                (approved, votes_for, votes.len() - votes_for)
            }
            VotingPolicy::SimpleMajority => {
                let votes_for = votes.iter().filter(|v| v.vote).count();
                let approved = votes_for > state.request.participants.len() / 2;
                (approved, votes_for, votes.len() - votes_for)
            }
            VotingPolicy::SuperMajority(threshold) => {
                let votes_for = votes.iter().filter(|v| v.vote).count();
                let required = (state.request.participants.len() as f64 * threshold) as usize;
                let approved = votes_for >= required;
                (approved, votes_for, votes.len() - votes_for)
            }
            VotingPolicy::Threshold(threshold) => {
                let votes_for = votes.iter().filter(|v| v.vote).count();
                let ratio = votes_for as f64 / state.request.participants.len() as f64;
                let approved = ratio >= threshold;
                (approved, votes_for, votes.len() - votes_for)
            }
            VotingPolicy::Weighted => {
                let total_weight: f64 = votes.iter().map(|v| v.weight).sum();
                let weight_for: f64 = votes.iter().filter(|v| v.vote).map(|v| v.weight).sum();
                let approved = weight_for > total_weight / 2.0;
                let votes_for = votes.iter().filter(|v| v.vote).count();
                (approved, votes_for, votes.len() - votes_for)
            }
        };
        
        let end_time = chrono::Utc::now();
        let completion_time_ms = (end_time - state.start_time).num_milliseconds() as u64;
        
        let result = ConsensusResult {
            request_id: state.request.id,
            approved,
            votes_for,
            votes_against,
            total_participants: state.request.participants.len(),
            participation_rate: votes.len() as f64 / state.request.participants.len() as f64,
            votes,
            completion_time_ms,
        };
        
        let _ = state.result_tx.send(result.clone());
        
        Ok(result)
    }

    /// Cancel consensus
    pub async fn cancel_consensus(&self, request_id: MessageId) -> anyhow::Result<()> {
        if let Some((_, state)) = self.active_consensus.remove(&request_id) {
            let result = ConsensusResult {
                request_id: state.request.id,
                approved: false,
                votes_for: 0,
                votes_against: 0,
                total_participants: state.request.participants.len(),
                participation_rate: 0.0,
                votes: vec![],
                completion_time_ms: 0,
            };
            
            let _ = state.result_tx.send(result);
        }
        
        Ok(())
    }

    /// Get active consensus count
    pub fn active_count(&self) -> usize {
        self.active_consensus.len()
    }

    /// Clean up completed consensus
    pub async fn cleanup(&self) {
        // Consensus entries are automatically removed when finalized
        // This method is here for any additional cleanup if needed
    }
}

/// Byzantine Fault Tolerant consensus implementation
pub struct BFTConsensus {
    manager: ConsensusManager,
    fault_tolerance: f64,
}

impl BFTConsensus {
    /// Create new BFT consensus with fault tolerance ratio
    pub fn new(fault_tolerance: f64, default_timeout_ms: u64) -> Self {
        Self {
            manager: ConsensusManager::new(default_timeout_ms),
            fault_tolerance,
        }
    }

    /// Calculate required votes for BFT consensus
    pub fn calculate_required_votes(&self, total_nodes: usize) -> usize {
        let byzantine_nodes = (total_nodes as f64 * self.fault_tolerance) as usize;
        total_nodes - byzantine_nodes
    }

    /// Initiate BFT consensus
    pub async fn initiate(
        &self,
        proposal: serde_json::Value,
        initiator: AgentId,
        participants: Vec<AgentId>,
        timeout_ms: u64,
    ) -> anyhow::Result<broadcast::Receiver<ConsensusResult>> {
        let required_ratio = 1.0 - self.fault_tolerance;
        
        let request = ConsensusRequest {
            id: MessageId::new(),
            proposal,
            policy: VotingPolicy::Threshold(required_ratio),
            timeout_ms,
            initiator,
            participants,
            metadata: HashMap::new(),
        };
        
        self.manager.initiate_consensus(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_majority_consensus() {
        let manager = ConsensusManager::new(5000);
        
        let initiator = AgentId::new();
        let participants = vec![
            AgentId::new(),
            AgentId::new(),
            AgentId::new(),
        ];
        
        let request = ConsensusRequest {
            id: MessageId::new(),
            proposal: serde_json::json!({"action": "increase_limit", "amount": 1000}),
            policy: VotingPolicy::SimpleMajority,
            timeout_ms: 1000,
            initiator,
            participants: participants.clone(),
            metadata: HashMap::new(),
        };
        
        let mut rx = manager.initiate_consensus(request.clone()).await.unwrap();
        
        // Submit votes
        for (i, &agent_id) in participants.iter().enumerate() {
            let vote = Vote {
                agent_id,
                vote: i < 2, // First two vote yes, last one no
                weight: 1.0,
                reason: None,
                timestamp: chrono::Utc::now(),
            };
            manager.submit_vote(vote).await.unwrap();
        }
        
        // Wait for result
        let result = rx.recv().await.unwrap();
        assert!(result.approved);
        assert_eq!(result.votes_for, 2);
        assert_eq!(result.votes_against, 1);
    }

    #[tokio::test]
    async fn test_unanimous_consensus() {
        let manager = ConsensusManager::new(5000);
        
        let participants = vec![AgentId::new(), AgentId::new()];
        
        let request = ConsensusRequest {
            id: MessageId::new(),
            proposal: serde_json::json!({"critical": true}),
            policy: VotingPolicy::Unanimous,
            timeout_ms: 1000,
            initiator: AgentId::new(),
            participants: participants.clone(),
            metadata: HashMap::new(),
        };
        
        let mut rx = manager.initiate_consensus(request).await.unwrap();
        
        // Submit votes - one yes, one no
        let vote1 = Vote {
            agent_id: participants[0],
            vote: true,
            weight: 1.0,
            reason: None,
            timestamp: chrono::Utc::now(),
        };
        manager.submit_vote(vote1).await.unwrap();
        
        let vote2 = Vote {
            agent_id: participants[1],
            vote: false,
            weight: 1.0,
            reason: Some("Too risky".to_string()),
            timestamp: chrono::Utc::now(),
        };
        manager.submit_vote(vote2).await.unwrap();
        
        let result = rx.recv().await.unwrap();
        assert!(!result.approved); // Should fail - not unanimous
    }

    #[tokio::test]
    async fn test_bft_consensus() {
        let bft = BFTConsensus::new(0.33, 5000); // Tolerate 1/3 byzantine nodes
        
        let participants: Vec<_> = (0..4).map(|_| AgentId::new()).collect();
        
        let mut rx = bft.initiate(
            serde_json::json!({"bft_test": true}),
            AgentId::new(),
            participants.clone(),
            1000,
        ).await.unwrap();
        
        // Need at least 3 out of 4 votes (67%)
        for i in 0..3 {
            let vote = Vote {
                agent_id: participants[i],
                vote: true,
                weight: 1.0,
                reason: None,
                timestamp: chrono::Utc::now(),
            };
            bft.manager.submit_vote(vote).await.unwrap();
        }
        
        // Fourth vote against
        let vote = Vote {
            agent_id: participants[3],
            vote: false,
            weight: 1.0,
            reason: None,
            timestamp: chrono::Utc::now(),
        };
        bft.manager.submit_vote(vote).await.unwrap();
        
        let result = rx.recv().await.unwrap();
        assert!(result.approved); // Should pass with 3/4 votes
    }
}