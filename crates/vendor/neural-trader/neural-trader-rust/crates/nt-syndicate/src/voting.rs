//! Voting system for syndicate governance

use crate::types::*;
use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Voting system for syndicate decisions
#[napi]
#[derive(Clone)]
pub struct VotingSystem {
    syndicate_id: String,
    active_votes: Arc<DashMap<String, VoteData>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VoteData {
    id: Uuid,
    proposal_type: String,
    details: serde_json::Value,
    proposed_by: Uuid,
    created_at: DateTime<Utc>,
    expires_at: DateTime<Utc>,
    status: String,
    votes: HashMap<String, CastVote>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CastVote {
    decision: String,
    weight: f64,
    timestamp: DateTime<Utc>,
}

#[napi]
impl VotingSystem {
    /// Create new voting system
    #[napi(constructor)]
    pub fn new(syndicate_id: String) -> Self {
        Self {
            syndicate_id,
            active_votes: Arc::new(DashMap::new()),
        }
    }

    /// Create a new vote
    #[napi]
    pub fn create_vote(
        &self,
        proposal_type: String,
        proposal_details: String,
        proposed_by: String,
        voting_period_hours: Option<i64>,
    ) -> napi::Result<String> {
        let details: serde_json::Value = serde_json::from_str(&proposal_details)
            .map_err(|e| napi::Error::from_reason(format!("Invalid proposal details: {}", e)))?;

        let proposer_id = Uuid::parse_str(&proposed_by)
            .map_err(|e| napi::Error::from_reason(format!("Invalid proposer ID: {}", e)))?;

        let vote_id = Uuid::new_v4();
        let period = voting_period_hours.unwrap_or(24);

        let vote_data = VoteData {
            id: vote_id,
            proposal_type: proposal_type.clone(),
            details,
            proposed_by: proposer_id,
            created_at: Utc::now(),
            expires_at: Utc::now() + Duration::hours(period),
            status: "active".to_string(),
            votes: HashMap::new(),
        };

        self.active_votes.insert(vote_id.to_string(), vote_data);

        Ok(vote_id.to_string())
    }

    /// Cast a vote
    #[napi]
    pub fn cast_vote(
        &self,
        vote_id: String,
        member_id: String,
        decision: String,
        voting_weight: f64,
    ) -> napi::Result<bool> {
        let mut vote = self.active_votes
            .get_mut(&vote_id)
            .ok_or_else(|| napi::Error::from_reason("Vote not found"))?;

        if vote.status != "active" {
            return Err(napi::Error::from_reason("Vote is not active"));
        }

        if Utc::now() > vote.expires_at {
            vote.status = "expired".to_string();
            return Err(napi::Error::from_reason("Vote has expired"));
        }

        // Validate decision
        if !["approve", "reject", "abstain"].contains(&decision.as_str()) {
            return Err(napi::Error::from_reason("Invalid decision. Must be 'approve', 'reject', or 'abstain'"));
        }

        let cast_vote = CastVote {
            decision,
            weight: voting_weight,
            timestamp: Utc::now(),
        };

        vote.votes.insert(member_id, cast_vote);

        Ok(true)
    }

    /// Get vote results
    #[napi]
    pub fn get_vote_results(&self, vote_id: String) -> napi::Result<String> {
        let vote = self.active_votes
            .get(&vote_id)
            .ok_or_else(|| napi::Error::from_reason("Vote not found"))?;

        let mut approve_weight = 0.0;
        let mut reject_weight = 0.0;
        let mut abstain_weight = 0.0;

        for cast_vote in vote.votes.values() {
            match cast_vote.decision.as_str() {
                "approve" => approve_weight += cast_vote.weight,
                "reject" => reject_weight += cast_vote.weight,
                "abstain" => abstain_weight += cast_vote.weight,
                _ => {}
            }
        }

        let total_weight = approve_weight + reject_weight + abstain_weight;
        let approval_percentage = if total_weight > 0.0 {
            (approve_weight / total_weight) * 100.0
        } else {
            0.0
        };

        let result = serde_json::json!({
            "vote_id": vote_id,
            "status": vote.status,
            "proposal_type": vote.proposal_type,
            "created_at": vote.created_at.to_rfc3339(),
            "expires_at": vote.expires_at.to_rfc3339(),
            "results": {
                "approve": approve_weight,
                "reject": reject_weight,
                "abstain": abstain_weight,
            },
            "total_votes": vote.votes.len(),
            "total_weight": total_weight,
            "approval_percentage": approval_percentage,
        });

        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Finalize vote
    #[napi]
    pub fn finalize_vote(&self, vote_id: String) -> napi::Result<String> {
        let mut vote = self.active_votes
            .get_mut(&vote_id)
            .ok_or_else(|| napi::Error::from_reason("Vote not found"))?;

        if vote.status != "active" {
            return Err(napi::Error::from_reason("Vote is not active"));
        }

        // Calculate results
        let mut approve_weight = 0.0;
        let mut reject_weight = 0.0;

        for cast_vote in vote.votes.values() {
            match cast_vote.decision.as_str() {
                "approve" => approve_weight += cast_vote.weight,
                "reject" => reject_weight += cast_vote.weight,
                _ => {}
            }
        }

        let total_weight = approve_weight + reject_weight;
        let approval_percentage = if total_weight > 0.0 {
            (approve_weight / total_weight) * 100.0
        } else {
            0.0
        };

        // Determine outcome (requires >50% approval)
        vote.status = if approval_percentage > 50.0 {
            "passed".to_string()
        } else {
            "failed".to_string()
        };

        let result = serde_json::json!({
            "vote_id": vote_id,
            "status": vote.status,
            "approval_percentage": approval_percentage,
            "passed": vote.status == "passed",
        });

        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// List all active votes
    #[napi]
    pub fn list_active_votes(&self) -> napi::Result<String> {
        let active: Vec<_> = self.active_votes
            .iter()
            .filter(|entry| entry.value().status == "active")
            .map(|entry| {
                let vote = entry.value();
                serde_json::json!({
                    "id": vote.id.to_string(),
                    "proposal_type": vote.proposal_type,
                    "created_at": vote.created_at.to_rfc3339(),
                    "expires_at": vote.expires_at.to_rfc3339(),
                    "votes_cast": vote.votes.len(),
                })
            })
            .collect();

        serde_json::to_string(&active)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Get vote details
    #[napi]
    pub fn get_vote_details(&self, vote_id: String) -> napi::Result<String> {
        let vote = self.active_votes
            .get(&vote_id)
            .ok_or_else(|| napi::Error::from_reason("Vote not found"))?;

        let result = serde_json::json!({
            "id": vote.id.to_string(),
            "proposal_type": vote.proposal_type,
            "details": vote.details,
            "proposed_by": vote.proposed_by.to_string(),
            "created_at": vote.created_at.to_rfc3339(),
            "expires_at": vote.expires_at.to_rfc3339(),
            "status": vote.status,
            "votes_cast": vote.votes.len(),
        });

        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Check if member has voted
    #[napi]
    pub fn has_voted(&self, vote_id: String, member_id: String) -> napi::Result<bool> {
        let vote = self.active_votes
            .get(&vote_id)
            .ok_or_else(|| napi::Error::from_reason("Vote not found"))?;

        Ok(vote.votes.contains_key(&member_id))
    }

    /// Get member's vote
    #[napi]
    pub fn get_member_vote(&self, vote_id: String, member_id: String) -> napi::Result<String> {
        let vote = self.active_votes
            .get(&vote_id)
            .ok_or_else(|| napi::Error::from_reason("Vote not found"))?;

        let member_vote = vote.votes.get(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member has not voted"))?;

        let result = serde_json::json!({
            "decision": member_vote.decision,
            "weight": member_vote.weight,
            "timestamp": member_vote.timestamp.to_rfc3339(),
        });

        serde_json::to_string(&result)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voting_system_creation() {
        let voting = VotingSystem::new("test-syndicate".to_string());
        assert_eq!(voting.syndicate_id, "test-syndicate");
    }

    #[test]
    fn test_create_vote() {
        let voting = VotingSystem::new("test".to_string());
        let details = serde_json::json!({"description": "Test proposal"}).to_string();
        let proposer = Uuid::new_v4().to_string();

        let result = voting.create_vote(
            "strategy_change".to_string(),
            details,
            proposer,
            Some(48),
        );

        assert!(result.is_ok());
    }
}
