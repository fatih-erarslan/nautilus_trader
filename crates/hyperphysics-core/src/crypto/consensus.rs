//! # Byzantine Consensus Module
//!
//! Implements Byzantine fault-tolerant consensus for multi-agent payment verification.
//! Uses threshold voting to achieve agreement among agents.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Byzantine fault-tolerant consensus coordinator
pub struct ByzantineConsensus {
    /// Consensus threshold (e.g., 2/3 = 0.6667)
    threshold: f64,
}

/// Agent vote on a proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Agent identifier
    pub agent_id: String,

    /// Vote decision (true = approve, false = reject)
    pub approve: bool,

    /// Hex-encoded Ed25519 signature
    pub signature: String,

    /// Hex-encoded public key of voter
    pub public_key: String,
}

/// Result of consensus calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    /// Whether consensus was achieved
    pub approved: bool,

    /// Number of votes in favor
    pub votes_for: usize,

    /// Number of votes against
    pub votes_against: usize,

    /// Total number of votes
    pub total: usize,

    /// Percentage of votes in favor
    pub approval_rate: f64,

    /// Whether threshold was met
    pub threshold_met: bool,
}

impl ByzantineConsensus {
    /// Create new Byzantine consensus coordinator
    ///
    /// # Arguments
    /// * `threshold` - Minimum fraction of approving votes (0.5 < threshold ≤ 1.0)
    ///
    /// # Example
    /// ```rust,no_run
    /// use hyperphysics_core::crypto::consensus::ByzantineConsensus;
    ///
    /// // Require 2/3 majority (67%)
    /// let consensus = ByzantineConsensus::new(2.0/3.0);
    /// ```
    pub fn new(threshold: f64) -> Self {
        assert!(
            threshold > 0.5 && threshold <= 1.0,
            "Threshold must be in range (0.5, 1.0]"
        );
        Self { threshold }
    }

    /// Check if consensus is reached on a set of votes
    ///
    /// # Arguments
    /// * `votes` - Collection of agent votes
    ///
    /// # Returns
    /// Consensus result with approval status and vote tallies
    pub fn check_consensus(&self, votes: &[Vote]) -> ConsensusResult {
        let total = votes.len();

        if total == 0 {
            return ConsensusResult {
                approved: false,
                votes_for: 0,
                votes_against: 0,
                total: 0,
                approval_rate: 0.0,
                threshold_met: false,
            };
        }

        let votes_for = votes.iter().filter(|v| v.approve).count();
        let votes_against = total - votes_for;
        let approval_rate = votes_for as f64 / total as f64;

        let threshold_met = approval_rate >= self.threshold;
        let approved = threshold_met && votes_for > votes_against;

        ConsensusResult {
            approved,
            votes_for,
            votes_against,
            total,
            approval_rate,
            threshold_met,
        }
    }

    /// Verify all vote signatures
    ///
    /// # Arguments
    /// * `votes` - Votes to verify
    /// * `message` - Original message that was voted on
    ///
    /// # Returns
    /// Result with list of invalid vote indices
    pub fn verify_votes(&self, votes: &[Vote], message: &[u8]) -> Result<(), Vec<usize>> {
        use ed25519_dalek::{VerifyingKey, Signature, Verifier};

        let mut invalid_indices = Vec::new();

        for (idx, vote) in votes.iter().enumerate() {
            // Decode public key
            let public_bytes = match hex::decode(&vote.public_key) {
                Ok(bytes) => bytes,
                Err(_) => {
                    invalid_indices.push(idx);
                    continue;
                }
            };

            let public_key_array: [u8; 32] = match public_bytes.try_into() {
                Ok(arr) => arr,
                Err(_) => {
                    invalid_indices.push(idx);
                    continue;
                }
            };

            let verifying_key = match VerifyingKey::from_bytes(&public_key_array) {
                Ok(key) => key,
                Err(_) => {
                    invalid_indices.push(idx);
                    continue;
                }
            };

            // Decode signature
            let sig_bytes = match hex::decode(&vote.signature) {
                Ok(bytes) => bytes,
                Err(_) => {
                    invalid_indices.push(idx);
                    continue;
                }
            };

            let sig_array: [u8; 64] = match sig_bytes.try_into() {
                Ok(arr) => arr,
                Err(_) => {
                    invalid_indices.push(idx);
                    continue;
                }
            };

            let signature = Signature::from_bytes(&sig_array);

            // Verify signature
            if verifying_key.verify(message, &signature).is_err() {
                invalid_indices.push(idx);
            }
        }

        if invalid_indices.is_empty() {
            Ok(())
        } else {
            Err(invalid_indices)
        }
    }

    /// Get current threshold
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Calculate minimum votes needed for approval
    pub fn min_votes_needed(&self, total_agents: usize) -> usize {
        (total_agents as f64 * self.threshold).ceil() as usize
    }
}

/// Multi-round consensus tracker
pub struct ConsensusTracker {
    consensus: ByzantineConsensus,
    rounds: HashMap<String, Vec<Vote>>,
}

impl ConsensusTracker {
    /// Create new consensus tracker
    pub fn new(threshold: f64) -> Self {
        Self {
            consensus: ByzantineConsensus::new(threshold),
            rounds: HashMap::new(),
        }
    }

    /// Add vote to a specific round
    pub fn add_vote(&mut self, round_id: String, vote: Vote) {
        self.rounds.entry(round_id).or_insert_with(Vec::new).push(vote);
    }

    /// Check consensus for a round
    pub fn check_round(&self, round_id: &str) -> Option<ConsensusResult> {
        self.rounds.get(round_id).map(|votes| {
            self.consensus.check_consensus(votes)
        })
    }

    /// Get all votes for a round
    pub fn get_votes(&self, round_id: &str) -> Option<&Vec<Vote>> {
        self.rounds.get(round_id)
    }

    /// Clear completed rounds
    pub fn clear_round(&mut self, round_id: &str) {
        self.rounds.remove(round_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::identity::AgentIdentity;

    #[test]
    fn test_consensus_threshold() {
        let consensus = ByzantineConsensus::new(0.67);
        assert_eq!(consensus.threshold(), 0.67);
        assert_eq!(consensus.min_votes_needed(3), 3);  // ceil(3 * 0.67) = 3
        assert_eq!(consensus.min_votes_needed(10), 7); // ceil(10 * 0.67) = 7
    }

    #[test]
    fn test_consensus_approval() {
        // Use 0.66 threshold so 2/3 approval (66.67%) passes
        let consensus = ByzantineConsensus::new(0.66);

        // Create mock votes (signatures not verified in this test)
        let votes = vec![
            Vote {
                agent_id: "agent1".to_string(),
                approve: true,
                signature: "sig1".to_string(),
                public_key: "pk1".to_string(),
            },
            Vote {
                agent_id: "agent2".to_string(),
                approve: true,
                signature: "sig2".to_string(),
                public_key: "pk2".to_string(),
            },
            Vote {
                agent_id: "agent3".to_string(),
                approve: false,
                signature: "sig3".to_string(),
                public_key: "pk3".to_string(),
            },
        ];

        let result = consensus.check_consensus(&votes);

        assert_eq!(result.votes_for, 2);
        assert_eq!(result.votes_against, 1);
        assert_eq!(result.total, 3);
        assert_eq!(result.approval_rate, 2.0/3.0);
        assert!(result.threshold_met);
        assert!(result.approved);
    }

    #[test]
    fn test_consensus_rejection() {
        let consensus = ByzantineConsensus::new(0.67);

        let votes = vec![
            Vote {
                agent_id: "agent1".to_string(),
                approve: true,
                signature: "sig1".to_string(),
                public_key: "pk1".to_string(),
            },
            Vote {
                agent_id: "agent2".to_string(),
                approve: false,
                signature: "sig2".to_string(),
                public_key: "pk2".to_string(),
            },
            Vote {
                agent_id: "agent3".to_string(),
                approve: false,
                signature: "sig3".to_string(),
                public_key: "pk3".to_string(),
            },
        ];

        let result = consensus.check_consensus(&votes);

        assert_eq!(result.votes_for, 1);
        assert_eq!(result.votes_against, 2);
        assert!(!result.threshold_met);
        assert!(!result.approved);
    }

    #[test]
    fn test_verify_votes_with_real_signatures() {
        let consensus = ByzantineConsensus::new(0.67);

        // Create real identities and sign a message
        let agent1 = AgentIdentity::generate("agent1".to_string());
        let agent2 = AgentIdentity::generate("agent2".to_string());

        let message = b"payment mandate authorization";

        let sig1 = agent1.sign(message).unwrap();
        let sig2 = agent2.sign(message).unwrap();

        let votes = vec![
            Vote {
                agent_id: agent1.agent_id().to_string(),
                approve: true,
                signature: hex::encode(sig1.to_bytes()),
                public_key: agent1.export_public_key(),
            },
            Vote {
                agent_id: agent2.agent_id().to_string(),
                approve: true,
                signature: hex::encode(sig2.to_bytes()),
                public_key: agent2.export_public_key(),
            },
        ];

        let result = consensus.verify_votes(&votes, message);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consensus_tracker() {
        let mut tracker = ConsensusTracker::new(0.67);

        let vote1 = Vote {
            agent_id: "agent1".to_string(),
            approve: true,
            signature: "sig1".to_string(),
            public_key: "pk1".to_string(),
        };

        tracker.add_vote("round1".to_string(), vote1);

        let votes = tracker.get_votes("round1").unwrap();
        assert_eq!(votes.len(), 1);

        tracker.clear_round("round1");
        assert!(tracker.get_votes("round1").is_none());
    }
}
