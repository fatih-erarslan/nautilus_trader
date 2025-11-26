//! Capital management for syndicates

use crate::{Error, Result};
use crate::models::{Member, ProfitDistribution};
use rust_decimal::Decimal;
use uuid::Uuid;
use std::collections::HashMap;

/// Capital manager for syndicate funds
pub struct CapitalManager {
    /// Total pooled capital
    total_capital: Decimal,
    /// Member balances
    member_balances: HashMap<Uuid, Decimal>,
    /// Profit distribution method
    distribution_method: ProfitDistribution,
}

impl CapitalManager {
    /// Create new capital manager
    pub fn new(distribution_method: ProfitDistribution) -> Self {
        Self {
            total_capital: Decimal::ZERO,
            member_balances: HashMap::new(),
            distribution_method,
        }
    }

    /// Add member's capital contribution
    pub fn add_contribution(&mut self, member_id: Uuid, amount: Decimal) -> Result<()> {
        if amount <= Decimal::ZERO {
            return Err(Error::ConfigError("Contribution must be positive".to_string()));
        }

        self.total_capital += amount;
        *self.member_balances.entry(member_id).or_insert(Decimal::ZERO) += amount;

        Ok(())
    }

    /// Withdraw member's capital
    pub fn withdraw(&mut self, member_id: Uuid, amount: Decimal) -> Result<()> {
        let balance = self.member_balances.get(&member_id)
            .ok_or_else(|| Error::MemberNotFound(member_id.to_string()))?;

        if amount > *balance {
            return Err(Error::InsufficientCapital {
                required: amount.to_string().parse().unwrap_or(0.0),
                available: balance.to_string().parse().unwrap_or(0.0),
            });
        }

        self.total_capital -= amount;
        *self.member_balances.get_mut(&member_id).unwrap() -= amount;

        Ok(())
    }

    /// Distribute profits to members
    pub fn distribute_profits(&mut self, profit: Decimal, members: &[Member]) -> Result<HashMap<Uuid, Decimal>> {
        let mut distributions = HashMap::new();

        match self.distribution_method {
            ProfitDistribution::Proportional => {
                // Distribute proportional to capital contributed
                for member in members.iter().filter(|m| m.is_active) {
                    if self.total_capital > Decimal::ZERO {
                        let share = member.capital_balance / self.total_capital;
                        let member_profit = profit * share;
                        distributions.insert(member.id, member_profit);
                        *self.member_balances.entry(member.id).or_insert(Decimal::ZERO) += member_profit;
                    }
                }
            }
            ProfitDistribution::Equal => {
                // Equal distribution among active members
                let active_count = members.iter().filter(|m| m.is_active).count();
                if active_count > 0 {
                    let share = profit / Decimal::from(active_count);
                    for member in members.iter().filter(|m| m.is_active) {
                        distributions.insert(member.id, share);
                        *self.member_balances.entry(member.id).or_insert(Decimal::ZERO) += share;
                    }
                }
            }
            ProfitDistribution::Performance => {
                // TODO: Implement performance-based distribution
                return Err(Error::Internal("Performance-based distribution not yet implemented".to_string()));
            }
        }

        self.total_capital += profit;
        Ok(distributions)
    }

    /// Get total capital
    pub fn total_capital(&self) -> Decimal {
        self.total_capital
    }

    /// Get member balance
    pub fn get_balance(&self, member_id: &Uuid) -> Option<Decimal> {
        self.member_balances.get(member_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capital_contribution() {
        let mut manager = CapitalManager::new(ProfitDistribution::Proportional);
        let member_id = Uuid::new_v4();

        assert!(manager.add_contribution(member_id, Decimal::from(5000)).is_ok());
        assert_eq!(manager.total_capital(), Decimal::from(5000));
        assert_eq!(manager.get_balance(&member_id), Some(Decimal::from(5000)));
    }

    #[test]
    fn test_withdraw() {
        let mut manager = CapitalManager::new(ProfitDistribution::Proportional);
        let member_id = Uuid::new_v4();

        manager.add_contribution(member_id, Decimal::from(5000)).unwrap();
        assert!(manager.withdraw(member_id, Decimal::from(2000)).is_ok());
        assert_eq!(manager.get_balance(&member_id), Some(Decimal::from(3000)));
    }
}
