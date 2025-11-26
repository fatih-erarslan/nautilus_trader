//! Voting system for syndicate governance

use crate::{Error, Result};
use crate::models::Member;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Proposal for syndicate vote
#[derive(Debug, Clone)]
pub struct Proposal {
    pub id: Uuid,
    pub title: String,
    pub description: String,
    pub proposer: Uuid,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub votes_for: HashMap<Uuid, f64>,  // Member ID -> voting weight
    pub votes_against: HashMap<Uuid, f64>,
    pub status: ProposalStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalStatus {
    Active,
    Passed,
    Rejected,
    Expired,
}

/// Voting system
pub struct VotingSystem {
    threshold: f64,  // 0.0 - 1.0 (percentage of votes needed to pass)
}

impl VotingSystem {
    /// Create new voting system
    pub fn new(threshold: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(Error::ConfigError("Threshold must be between 0 and 1".to_string()));
        }

        Ok(Self { threshold })
    }

    /// Cast a vote
    pub fn vote(&self, proposal: &mut Proposal, member: &Member, vote_for: bool) -> Result<()> {
        if !member.can_vote() {
            return Err(Error::InsufficientVotingPower {
                required: 0.0,
                actual: 0.0,
            });
        }

        if vote_for {
            proposal.votes_for.insert(member.id, member.voting_weight);
            proposal.votes_against.remove(&member.id);
        } else {
            proposal.votes_against.insert(member.id, member.voting_weight);
            proposal.votes_for.remove(&member.id);
        }

        Ok(())
    }

    /// Check if proposal has passed
    pub fn check_result(&self, proposal: &Proposal, total_voting_weight: f64) -> ProposalStatus {
        if Utc::now() > proposal.expires_at {
            return ProposalStatus::Expired;
        }

        let votes_for: f64 = proposal.votes_for.values().sum();
        let votes_against: f64 = proposal.votes_against.values().sum();

        let support_percentage = if total_voting_weight > 0.0 {
            votes_for / total_voting_weight
        } else {
            0.0
        };

        if support_percentage >= self.threshold {
            ProposalStatus::Passed
        } else if (votes_against / total_voting_weight) > (1.0 - self.threshold) {
            ProposalStatus::Rejected
        } else {
            ProposalStatus::Active
        }
    }
}
