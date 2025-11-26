//! Data models for sports betting

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Member role in the syndicate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    /// Syndicate owner with full control
    Owner,
    /// Admin with management permissions
    Admin,
    /// Active member with voting rights
    Member,
    /// Observer with read-only access
    Observer,
}

/// Syndicate member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    /// Unique member ID
    pub id: Uuid,
    /// Member name or username
    pub name: String,
    /// Member role
    pub role: MemberRole,
    /// Capital contributed
    pub capital_contributed: Decimal,
    /// Current capital balance
    pub capital_balance: Decimal,
    /// Voting weight (0.0 - 1.0)
    pub voting_weight: f64,
    /// Member since timestamp
    pub joined_at: DateTime<Utc>,
    /// Last active timestamp
    pub last_active: DateTime<Utc>,
    /// Is member active
    pub is_active: bool,
}

/// Syndicate configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndicateConfig {
    /// Syndicate name
    pub name: String,
    /// Minimum capital contribution
    pub min_contribution: Decimal,
    /// Maximum number of members
    pub max_members: usize,
    /// Voting threshold for decisions (0.0 - 1.0)
    pub voting_threshold: f64,
    /// Profit distribution method
    pub profit_distribution: ProfitDistribution,
    /// Risk parameters
    pub max_bet_size: Decimal,
    /// Maximum total exposure
    pub max_total_exposure: Decimal,
}

/// Profit distribution method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfitDistribution {
    /// Proportional to capital contributed
    Proportional,
    /// Equal distribution among active members
    Equal,
    /// Custom distribution based on performance
    Performance,
}

/// Bet position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetPosition {
    /// Unique position ID
    pub id: Uuid,
    /// Sport type
    pub sport: String,
    /// Event description
    pub event: String,
    /// Bet type (moneyline, spread, over/under, etc.)
    pub bet_type: String,
    /// Selection
    pub selection: String,
    /// Odds at placement
    pub odds: Decimal,
    /// Stake amount
    pub stake: Decimal,
    /// Potential payout
    pub potential_payout: Decimal,
    /// Position status
    pub status: BetStatus,
    /// Placement timestamp
    pub placed_at: DateTime<Utc>,
    /// Settlement timestamp
    pub settled_at: Option<DateTime<Utc>>,
    /// Actual payout (if settled)
    pub actual_payout: Option<Decimal>,
}

/// Bet status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BetStatus {
    /// Bet is pending placement
    Pending,
    /// Bet is active
    Active,
    /// Bet won
    Won,
    /// Bet lost
    Lost,
    /// Bet pushed (tie)
    Push,
    /// Bet cancelled
    Cancelled,
}

/// Risk metrics for portfolio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Total capital at risk
    pub total_exposure: Decimal,
    /// Largest single bet
    pub max_bet_size: Decimal,
    /// Number of active bets
    pub active_bets: usize,
    /// Current win rate
    pub win_rate: f64,
    /// Average odds
    pub avg_odds: f64,
    /// Portfolio variance
    pub variance: f64,
    /// Expected value
    pub expected_value: Decimal,
    /// Kelly criterion recommendation
    pub kelly_fraction: f64,
}

impl Default for SyndicateConfig {
    fn default() -> Self {
        Self {
            name: "default-syndicate".to_string(),
            min_contribution: Decimal::from(1000),
            max_members: 10,
            voting_threshold: 0.66,
            profit_distribution: ProfitDistribution::Proportional,
            max_bet_size: Decimal::from(1000),
            max_total_exposure: Decimal::from(10000),
        }
    }
}

impl Member {
    /// Create a new member
    pub fn new(name: String, role: MemberRole, capital: Decimal) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            role,
            capital_contributed: capital,
            capital_balance: capital,
            voting_weight: 0.0,
            joined_at: Utc::now(),
            last_active: Utc::now(),
            is_active: true,
        }
    }

    /// Check if member has admin privileges
    pub fn is_admin(&self) -> bool {
        matches!(self.role, MemberRole::Owner | MemberRole::Admin)
    }

    /// Check if member can vote
    pub fn can_vote(&self) -> bool {
        self.is_active && matches!(self.role, MemberRole::Owner | MemberRole::Admin | MemberRole::Member)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_member_creation() {
        let member = Member::new(
            "alice".to_string(),
            MemberRole::Member,
            Decimal::from(5000),
        );

        assert_eq!(member.name, "alice");
        assert_eq!(member.role, MemberRole::Member);
        assert_eq!(member.capital_contributed, Decimal::from(5000));
        assert!(member.is_active);
        assert!(member.can_vote());
    }

    #[test]
    fn test_admin_privileges() {
        let owner = Member::new("owner".to_string(), MemberRole::Owner, Decimal::from(10000));
        let admin = Member::new("admin".to_string(), MemberRole::Admin, Decimal::from(5000));
        let member = Member::new("member".to_string(), MemberRole::Member, Decimal::from(1000));

        assert!(owner.is_admin());
        assert!(admin.is_admin());
        assert!(!member.is_admin());
    }

    #[test]
    fn test_voting_rights() {
        let member = Member::new("voter".to_string(), MemberRole::Member, Decimal::from(1000));
        let observer = Member::new("observer".to_string(), MemberRole::Observer, Decimal::from(0));

        assert!(member.can_vote());
        assert!(!observer.can_vote());
    }
}
