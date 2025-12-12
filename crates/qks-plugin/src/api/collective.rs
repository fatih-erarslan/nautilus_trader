//! # Layer 5: Collective Intelligence API
//!
//! Multi-agent swarm coordination and distributed consensus.
//!
//! ## Scientific Foundation
//!
//! **Swarm Intelligence**:
//! - Emergent collective behavior from simple local rules
//! - Self-organization without central control
//! - Stigmergy: Indirect coordination through environment
//!
//! **Consensus Algorithms**:
//! - **Raft**: Leader-based consensus for fault tolerance
//! - **Byzantine**: Tolerates arbitrary failures (f < n/3)
//! - **CRDT**: Conflict-free replicated data types
//!
//! ## Key Concepts
//!
//! ```text
//! Swarm Coordination:
//!   x_i(t+1) = x_i(t) + v_i(t)
//!   v_i(t+1) = w·v_i(t) + c1·r1·(p_i - x_i) + c2·r2·(g - x_i)
//!
//!   where:
//!   x_i = position of agent i
//!   v_i = velocity of agent i
//!   p_i = personal best
//!   g = global best
//! ```

use crate::{Result, QksError};
use std::collections::HashMap;
use uuid::Uuid;

/// Minimum quorum size for consensus
pub const MIN_QUORUM_SIZE: usize = 3;

/// Byzantine fault tolerance threshold (f < n/3)
pub const BYZANTINE_THRESHOLD_RATIO: f64 = 1.0 / 3.0;

/// Agent in the swarm
#[derive(Debug, Clone)]
pub struct Agent {
    /// Unique agent identifier
    pub id: String,
    /// Current state/position
    pub state: Vec<f64>,
    /// Agent role
    pub role: AgentRole,
    /// Trust score (0-1)
    pub trust: f64,
    /// Last heartbeat time
    pub last_heartbeat: f64,
}

/// Agent role in swarm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    /// Leader/coordinator
    Leader,
    /// Follower/worker
    Follower,
    /// Observer (read-only)
    Observer,
}

/// Consensus proposal
#[derive(Debug, Clone)]
pub struct Proposal {
    /// Proposal ID
    pub id: String,
    /// Proposer agent ID
    pub proposer: String,
    /// Proposal content
    pub content: Vec<f64>,
    /// Votes received
    pub votes: HashMap<String, Vote>,
    /// Consensus status
    pub status: ConsensusStatus,
}

/// Vote on a proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vote {
    /// Accept proposal
    Accept,
    /// Reject proposal
    Reject,
    /// Abstain
    Abstain,
}

/// Consensus status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsensusStatus {
    /// Proposal pending
    Pending,
    /// Consensus achieved
    Achieved,
    /// Consensus failed
    Failed,
    /// Timeout
    Timeout,
}

/// Swarm state
#[derive(Debug, Clone)]
pub struct SwarmState {
    /// All agents in swarm
    pub agents: HashMap<String, Agent>,
    /// Active proposals
    pub proposals: HashMap<String, Proposal>,
    /// Global best position
    pub global_best: Vec<f64>,
    /// Swarm cohesion (0-1)
    pub cohesion: f64,
}

impl SwarmState {
    /// Create new empty swarm
    pub fn new() -> Self {
        Self {
            agents: HashMap::new(),
            proposals: HashMap::new(),
            global_best: vec![],
            cohesion: 0.0,
        }
    }

    /// Get number of agents
    pub fn size(&self) -> usize {
        self.agents.len()
    }

    /// Check if agent is leader
    pub fn is_leader(&self, agent_id: &str) -> bool {
        self.agents
            .get(agent_id)
            .map(|a| a.role == AgentRole::Leader)
            .unwrap_or(false)
    }

    /// Get leader agent
    pub fn get_leader(&self) -> Option<&Agent> {
        self.agents.values().find(|a| a.role == AgentRole::Leader)
    }
}

impl Default for SwarmState {
    fn default() -> Self {
        Self::new()
    }
}

/// Join the swarm
///
/// # Arguments
/// * `agent_id` - Unique agent identifier
/// * `initial_state` - Initial agent state
/// * `role` - Agent role in swarm
///
/// # Returns
/// Success indicator
///
/// # Example
/// ```rust,ignore
/// join_swarm("agent_001", vec![0.0, 0.0], AgentRole::Follower)?;
/// ```
pub fn join_swarm(agent_id: &str, initial_state: Vec<f64>, role: AgentRole) -> Result<()> {
    // TODO: Interface with swarm coordinator (dilithium-mcp)
    Ok(())
}

/// Leave the swarm gracefully
///
/// # Arguments
/// * `agent_id` - Agent identifier
///
/// # Example
/// ```rust,ignore
/// leave_swarm("agent_001")?;
/// ```
pub fn leave_swarm(agent_id: &str) -> Result<()> {
    // TODO: Interface with swarm coordinator
    Ok(())
}

/// Send message to other agents
///
/// # Arguments
/// * `sender_id` - Sender agent ID
/// * `recipient_id` - Recipient agent ID (or "broadcast" for all)
/// * `message` - Message content
///
/// # Example
/// ```rust,ignore
/// send_message("agent_001", "agent_002", vec![1.0, 2.0])?;
/// send_message("agent_001", "broadcast", vec![0.5])?; // Broadcast
/// ```
pub fn send_message(sender_id: &str, recipient_id: &str, message: Vec<f64>) -> Result<()> {
    // TODO: Interface with message queue
    Ok(())
}

/// Receive messages for agent
///
/// # Arguments
/// * `agent_id` - Agent identifier
///
/// # Returns
/// Vector of received messages
pub fn receive_messages(agent_id: &str) -> Result<Vec<(String, Vec<f64>)>> {
    // TODO: Interface with message queue
    Ok(vec![])
}

/// Create consensus proposal
///
/// # Arguments
/// * `proposer_id` - Proposer agent ID
/// * `content` - Proposal content
///
/// # Returns
/// Proposal ID
///
/// # Example
/// ```rust,ignore
/// let proposal_id = create_proposal("agent_001", vec![1.0, 2.0])?;
/// ```
pub fn create_proposal(proposer_id: &str, content: Vec<f64>) -> Result<String> {
    let proposal_id = Uuid::new_v4().to_string();

    // TODO: Store proposal in distributed state
    Ok(proposal_id)
}

/// Vote on a proposal
///
/// # Arguments
/// * `agent_id` - Voting agent ID
/// * `proposal_id` - Proposal to vote on
/// * `vote` - Vote decision
///
/// # Example
/// ```rust,ignore
/// vote("agent_001", &proposal_id, Vote::Accept)?;
/// ```
pub fn vote(agent_id: &str, proposal_id: &str, vote: Vote) -> Result<()> {
    // TODO: Record vote and check for consensus
    Ok(())
}

/// Check if consensus achieved
///
/// # Arguments
/// * `proposal_id` - Proposal ID
///
/// # Returns
/// Consensus status and final decision
///
/// # Example
/// ```rust,ignore
/// let (status, accepted) = check_consensus(&proposal_id)?;
/// if status == ConsensusStatus::Achieved {
///     println!("Consensus reached!");
/// }
/// ```
pub fn check_consensus(proposal_id: &str) -> Result<(ConsensusStatus, bool)> {
    // TODO: Compute consensus based on votes
    Ok((ConsensusStatus::Pending, false))
}

/// Achieve distributed consensus using specified algorithm
///
/// # Arguments
/// * `agents` - Participating agents
/// * `proposal` - Proposal to reach consensus on
/// * `algorithm` - Consensus algorithm to use
///
/// # Returns
/// Consensus result
pub fn achieve_consensus(
    agents: &[String],
    proposal: &Proposal,
    algorithm: ConsensusAlgorithm,
) -> Result<bool> {
    match algorithm {
        ConsensusAlgorithm::Majority => majority_consensus(agents, proposal),
        ConsensusAlgorithm::Unanimous => unanimous_consensus(agents, proposal),
        ConsensusAlgorithm::Byzantine => byzantine_consensus(agents, proposal),
        ConsensusAlgorithm::Raft => raft_consensus(agents, proposal),
    }
}

/// Consensus algorithm type
#[derive(Debug, Clone, Copy)]
pub enum ConsensusAlgorithm {
    /// Simple majority (> 50%)
    Majority,
    /// Unanimous agreement
    Unanimous,
    /// Byzantine fault tolerant
    Byzantine,
    /// Raft leader-based
    Raft,
}

/// Majority consensus (> 50% accept)
fn majority_consensus(agents: &[String], proposal: &Proposal) -> Result<bool> {
    let accept_count = proposal
        .votes
        .values()
        .filter(|&&v| v == Vote::Accept)
        .count();

    let total_votes = proposal.votes.len();

    Ok(accept_count > total_votes / 2)
}

/// Unanimous consensus (all accept)
fn unanimous_consensus(agents: &[String], proposal: &Proposal) -> Result<bool> {
    Ok(proposal.votes.values().all(|&v| v == Vote::Accept))
}

/// Byzantine consensus (tolerates f < n/3 faults)
fn byzantine_consensus(agents: &[String], proposal: &Proposal) -> Result<bool> {
    let n = agents.len();
    let max_faults = (n as f64 * BYZANTINE_THRESHOLD_RATIO).floor() as usize;

    let accept_count = proposal
        .votes
        .values()
        .filter(|&&v| v == Vote::Accept)
        .count();

    // Need 2f + 1 accepts for Byzantine consensus
    Ok(accept_count >= 2 * max_faults + 1)
}

/// Raft consensus (leader-based)
fn raft_consensus(agents: &[String], proposal: &Proposal) -> Result<bool> {
    // Simplified Raft: Leader decision + majority confirmation
    majority_consensus(agents, proposal)
}

/// Compute swarm cohesion (how tightly clustered)
///
/// # Arguments
/// * `agent_states` - States of all agents
///
/// # Returns
/// Cohesion measure (0-1, higher = more cohesive)
pub fn swarm_cohesion(agent_states: &[Vec<f64>]) -> f64 {
    if agent_states.len() < 2 {
        return 1.0;
    }

    // Compute pairwise distances
    let mut total_distance = 0.0;
    let mut count = 0;

    for i in 0..agent_states.len() {
        for j in (i + 1)..agent_states.len() {
            total_distance += euclidean_distance(&agent_states[i], &agent_states[j]);
            count += 1;
        }
    }

    let avg_distance = total_distance / count as f64;

    // Convert to cohesion (inverse of distance)
    // Use sigmoid to map to [0,1]
    1.0 / (1.0 + avg_distance)
}

/// Euclidean distance between two vectors
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Update agent state using swarm dynamics
///
/// # Arguments
/// * `agent_state` - Current agent state
/// * `personal_best` - Agent's personal best
/// * `global_best` - Swarm global best
/// * `inertia` - Inertia weight
/// * `cognitive` - Cognitive coefficient
/// * `social` - Social coefficient
///
/// # Returns
/// Updated agent state
pub fn update_swarm_position(
    agent_state: &[f64],
    personal_best: &[f64],
    global_best: &[f64],
    inertia: f64,
    cognitive: f64,
    social: f64,
) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    use rand::Rng;

    agent_state
        .iter()
        .zip(personal_best.iter())
        .zip(global_best.iter())
        .map(|((x, p), g)| {
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();

            // Particle swarm update rule
            inertia * x + cognitive * r1 * (p - x) + social * r2 * (g - x)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_state() {
        let swarm = SwarmState::new();
        assert_eq!(swarm.size(), 0);
    }

    #[test]
    fn test_majority_consensus() {
        let agents = vec!["a1".to_string(), "a2".to_string(), "a3".to_string()];

        let mut proposal = Proposal {
            id: "p1".to_string(),
            proposer: "a1".to_string(),
            content: vec![],
            votes: HashMap::new(),
            status: ConsensusStatus::Pending,
        };

        proposal.votes.insert("a1".to_string(), Vote::Accept);
        proposal.votes.insert("a2".to_string(), Vote::Accept);
        proposal.votes.insert("a3".to_string(), Vote::Reject);

        let result = majority_consensus(&agents, &proposal).unwrap();
        assert!(result); // 2/3 accept
    }

    #[test]
    fn test_unanimous_consensus() {
        let agents = vec!["a1".to_string(), "a2".to_string()];

        let mut proposal = Proposal {
            id: "p1".to_string(),
            proposer: "a1".to_string(),
            content: vec![],
            votes: HashMap::new(),
            status: ConsensusStatus::Pending,
        };

        proposal.votes.insert("a1".to_string(), Vote::Accept);
        proposal.votes.insert("a2".to_string(), Vote::Accept);

        let result = unanimous_consensus(&agents, &proposal).unwrap();
        assert!(result);
    }

    #[test]
    fn test_swarm_cohesion() {
        let states = vec![vec![0.0, 0.0], vec![0.1, 0.1], vec![0.05, 0.05]];

        let cohesion = swarm_cohesion(&states);
        assert!(cohesion > 0.5); // Tightly clustered
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-10);
    }
}
