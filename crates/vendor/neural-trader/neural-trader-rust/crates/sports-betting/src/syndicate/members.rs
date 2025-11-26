//! Member management

use crate::{Error, Result};
use crate::models::{Member, MemberRole};
use dashmap::DashMap;
use uuid::Uuid;
use rust_decimal::Decimal;

/// Member manager
pub struct MemberManager {
    members: DashMap<Uuid, Member>,
    max_members: usize,
}

impl MemberManager {
    /// Create new member manager
    pub fn new(max_members: usize) -> Self {
        Self {
            members: DashMap::new(),
            max_members,
        }
    }

    /// Add a new member
    pub fn add_member(&self, name: String, role: MemberRole, capital: Decimal) -> Result<Uuid> {
        if self.members.len() >= self.max_members {
            return Err(Error::ConfigError(format!("Maximum members ({}) reached", self.max_members)));
        }

        let member = Member::new(name, role, capital);
        let id = member.id;
        self.members.insert(id, member);

        Ok(id)
    }

    /// Get member by ID
    pub fn get_member(&self, id: &Uuid) -> Option<Member> {
        self.members.get(id).map(|m| m.clone())
    }

    /// Remove member
    pub fn remove_member(&self, id: &Uuid) -> Result<()> {
        self.members.remove(id)
            .ok_or_else(|| Error::MemberNotFound(id.to_string()))?;
        Ok(())
    }

    /// Get all active members
    pub fn get_active_members(&self) -> Vec<Member> {
        self.members.iter()
            .map(|entry| entry.value().clone())
            .filter(|m| m.is_active)
            .collect()
    }

    /// Update voting weights based on capital
    pub fn update_voting_weights(&self) {
        let total_capital: Decimal = self.members.iter()
            .filter(|entry| entry.value().is_active)
            .map(|entry| entry.value().capital_balance)
            .sum();

        if total_capital > Decimal::ZERO {
            for mut entry in self.members.iter_mut() {
                let weight = (entry.value().capital_balance / total_capital)
                    .to_string()
                    .parse::<f64>()
                    .unwrap_or(0.0);
                entry.voting_weight = weight;
            }
        }
    }
}
