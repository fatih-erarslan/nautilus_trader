//! High-level syndicate manager combining all components

use crate::{Error, Result};
use crate::models::{Member, MemberRole, SyndicateConfig};
use crate::syndicate::{CapitalManager, VotingSystem, MemberManager};
use rust_decimal::Decimal;
use uuid::Uuid;
use parking_lot::RwLock;
use std::sync::Arc;

/// Complete syndicate manager
pub struct SyndicateManager {
    config: Arc<RwLock<SyndicateConfig>>,
    capital: Arc<RwLock<CapitalManager>>,
    voting: Arc<VotingSystem>,
    members: Arc<MemberManager>,
}

impl SyndicateManager {
    /// Create new syndicate
    pub fn new(name: String) -> Self {
        let config = SyndicateConfig {
            name,
            ..Default::default()
        };

        let capital = CapitalManager::new(config.profit_distribution);
        let voting = VotingSystem::new(config.voting_threshold).expect("Valid threshold");
        let members = MemberManager::new(config.max_members);

        Self {
            config: Arc::new(RwLock::new(config)),
            capital: Arc::new(RwLock::new(capital)),
            voting: Arc::new(voting),
            members: Arc::new(members),
        }
    }

    /// Add a new member with capital contribution
    pub async fn add_member(&self, name: &str, capital_amount: f64) -> Result<Uuid> {
        let capital_decimal = Decimal::from_f64_retain(capital_amount)
            .ok_or_else(|| Error::Internal("Invalid capital amount".to_string()))?;

        // Check minimum contribution
        let min_contribution = self.config.read().min_contribution;
        if capital_decimal < min_contribution {
            return Err(Error::InsufficientCapital {
                required: min_contribution.to_string().parse().unwrap_or(0.0),
                available: capital_amount,
            });
        }

        // Add member
        let member_id = self.members.add_member(
            name.to_string(),
            MemberRole::Member,
            capital_decimal,
        )?;

        // Add capital contribution
        self.capital.write().add_contribution(member_id, capital_decimal)?;

        // Update voting weights
        self.members.update_voting_weights();

        Ok(member_id)
    }

    /// Get total syndicate capital
    pub fn total_capital(&self) -> Decimal {
        self.capital.read().total_capital()
    }

    /// Get all active members
    pub fn get_active_members(&self) -> Vec<Member> {
        self.members.get_active_members()
    }

    /// Distribute profits
    pub fn distribute_profits(&self, profit: Decimal) -> Result<()> {
        let members = self.get_active_members();
        self.capital.write().distribute_profits(profit, &members)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_syndicate_creation() {
        let syndicate = SyndicateManager::new("test-syndicate".to_string());
        assert_eq!(syndicate.total_capital(), Decimal::ZERO);
    }

    #[tokio::test]
    async fn test_add_member() {
        let syndicate = SyndicateManager::new("test-syndicate".to_string());
        let member_id = syndicate.add_member("alice", 5000.0).await.unwrap();

        let members = syndicate.get_active_members();
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].id, member_id);
        assert_eq!(syndicate.total_capital(), Decimal::from(5000));
    }
}
