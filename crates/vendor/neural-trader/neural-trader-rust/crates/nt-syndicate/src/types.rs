//! Core types for syndicate management

use chrono::{DateTime, Duration, Utc};
use napi_derive::napi;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Fund allocation strategies
#[napi]
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Kelly Criterion with fractional betting
    KellyCriterion,
    /// Fixed percentage allocation
    FixedPercentage,
    /// Dynamic confidence-based allocation
    DynamicConfidence,
    /// Risk parity allocation
    RiskParity,
    /// Martingale strategy
    Martingale,
    /// Anti-martingale strategy
    AntiMartingale,
}

/// Profit distribution models
#[napi]
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistributionModel {
    /// Pure proportional distribution
    Proportional,
    /// Performance-weighted distribution
    PerformanceWeighted,
    /// Tiered distribution
    Tiered,
    /// Hybrid distribution (50% capital, 30% performance, 20% equal)
    Hybrid,
}

/// Member roles within the syndicate
#[napi]
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    /// Lead investor with full control
    LeadInvestor,
    /// Senior analyst with advanced permissions
    SeniorAnalyst,
    /// Junior analyst
    JuniorAnalyst,
    /// Contributing member
    ContributingMember,
    /// Observer with limited permissions
    Observer,
}

/// Investment tiers based on capital contribution
#[napi]
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberTier {
    /// Bronze tier ($1,000 - $5,000)
    Bronze,
    /// Silver tier ($5,000 - $25,000)
    Silver,
    /// Gold tier ($25,000 - $100,000)
    Gold,
    /// Platinum tier ($100,000+)
    Platinum,
}

/// Bankroll management rules
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankrollRules {
    /// Maximum single bet as percentage of total bankroll (default: 0.05 = 5%)
    pub max_single_bet: f64,
    /// Maximum daily exposure (default: 0.20 = 20%)
    pub max_daily_exposure: f64,
    /// Maximum sport concentration (default: 0.40 = 40%)
    pub max_sport_concentration: f64,
    /// Minimum reserve (default: 0.30 = 30%)
    pub minimum_reserve: f64,
    /// Daily stop loss limit (default: 0.10 = 10%)
    pub stop_loss_daily: f64,
    /// Weekly stop loss limit (default: 0.20 = 20%)
    pub stop_loss_weekly: f64,
    /// Profit lock percentage (default: 0.50 = 50%)
    pub profit_lock: f64,
    /// Maximum parlay percentage (default: 0.02 = 2%)
    pub max_parlay_percentage: f64,
    /// Maximum live betting percentage (default: 0.15 = 15%)
    pub max_live_betting: f64,
}

impl Default for BankrollRules {
    fn default() -> Self {
        Self {
            max_single_bet: 0.05,
            max_daily_exposure: 0.20,
            max_sport_concentration: 0.40,
            minimum_reserve: 0.30,
            stop_loss_daily: 0.10,
            stop_loss_weekly: 0.20,
            profit_lock: 0.50,
            max_parlay_percentage: 0.02,
            max_live_betting: 0.15,
        }
    }
}

/// Member permissions configuration
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberPermissions {
    /// Can create syndicates
    pub create_syndicate: bool,
    /// Can modify strategy
    pub modify_strategy: bool,
    /// Can approve large bets
    pub approve_large_bets: bool,
    /// Can manage members
    pub manage_members: bool,
    /// Can distribute profits
    pub distribute_profits: bool,
    /// Can access all analytics
    pub access_all_analytics: bool,
    /// Has veto power
    pub veto_power: bool,
    /// Can propose bets
    pub propose_bets: bool,
    /// Can access advanced analytics
    pub access_advanced_analytics: bool,
    /// Can create models
    pub create_models: bool,
    /// Can vote on strategy
    pub vote_on_strategy: bool,
    /// Can manage junior analysts
    pub manage_junior_analysts: bool,
    /// Can view bets
    pub view_bets: bool,
    /// Can vote on major decisions
    pub vote_on_major_decisions: bool,
    /// Can access basic analytics
    pub access_basic_analytics: bool,
    /// Can propose ideas
    pub propose_ideas: bool,
    /// Can withdraw own funds
    pub withdraw_own_funds: bool,
    /// Can create votes
    pub create_votes: bool,
}

impl Default for MemberPermissions {
    fn default() -> Self {
        Self {
            create_syndicate: false,
            modify_strategy: false,
            approve_large_bets: false,
            manage_members: false,
            distribute_profits: false,
            access_all_analytics: false,
            veto_power: false,
            propose_bets: false,
            access_advanced_analytics: false,
            create_models: false,
            vote_on_strategy: false,
            manage_junior_analysts: false,
            view_bets: true,
            vote_on_major_decisions: false,
            access_basic_analytics: true,
            propose_ideas: true,
            withdraw_own_funds: true,
            create_votes: false,
        }
    }
}

impl MemberPermissions {
    /// Get permissions for a given role
    pub fn for_role(role: MemberRole) -> Self {
        match role {
            MemberRole::LeadInvestor => Self {
                create_syndicate: true,
                modify_strategy: true,
                approve_large_bets: true,
                manage_members: true,
                distribute_profits: true,
                access_all_analytics: true,
                veto_power: true,
                propose_bets: true,
                vote_on_strategy: true,
                vote_on_major_decisions: true,
                create_votes: true,
                ..Default::default()
            },
            MemberRole::SeniorAnalyst => Self {
                propose_bets: true,
                access_advanced_analytics: true,
                create_models: true,
                vote_on_strategy: true,
                manage_junior_analysts: true,
                vote_on_major_decisions: true,
                ..Default::default()
            },
            MemberRole::JuniorAnalyst => Self {
                propose_bets: true,
                access_advanced_analytics: true,
                ..Default::default()
            },
            MemberRole::ContributingMember => Self {
                vote_on_major_decisions: true,
                access_basic_analytics: true,
                propose_ideas: true,
                withdraw_own_funds: true,
                ..Default::default()
            },
            MemberRole::Observer => Self {
                view_bets: true,
                access_basic_analytics: false,
                propose_ideas: false,
                withdraw_own_funds: false,
                ..Default::default()
            },
        }
    }
}

/// Betting opportunity details
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BettingOpportunity {
    /// Sport type
    pub sport: String,
    /// Event description
    pub event: String,
    /// Bet type (moneyline, spread, total, etc.)
    pub bet_type: String,
    /// Selection
    pub selection: String,
    /// Decimal odds
    pub odds: f64,
    /// Estimated probability
    pub probability: f64,
    /// Betting edge
    pub edge: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
    /// Model agreement (0-1)
    pub model_agreement: f64,
    /// Time until event in seconds
    pub time_until_event_secs: i64,
    /// Market liquidity
    pub liquidity: f64,
    /// Is live betting
    pub is_live: bool,
    /// Is parlay bet
    pub is_parlay: bool,
}

/// Allocation result
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationResult {
    /// Allocated amount
    pub amount: String,
    /// Percentage of bankroll
    pub percentage_of_bankroll: f64,
    /// Reasoning for allocation
    pub reasoning: String,
    /// Risk metrics (JSON serialized)
    pub risk_metrics: String,
    /// Approval required
    pub approval_required: bool,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Recommended stake sizing options (JSON serialized)
    pub recommended_stake_sizing: String,
}

/// Member statistics
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemberStatistics {
    /// Bets proposed
    pub bets_proposed: i64,
    /// Bets won
    pub bets_won: i64,
    /// Bets lost
    pub bets_lost: i64,
    /// Total profit
    pub total_profit: String,
    /// Total staked
    pub total_staked: String,
    /// ROI percentage
    pub roi: f64,
    /// Win rate
    pub win_rate: f64,
    /// Accuracy score
    pub accuracy: f64,
    /// Profit contribution
    pub profit_contribution: String,
    /// Votes cast
    pub votes_cast: i64,
    /// Strategy contributions
    pub strategy_contributions: i64,
}

/// Exposure tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureTracking {
    /// Daily exposure
    pub daily: Decimal,
    /// Weekly exposure
    pub weekly: Decimal,
    /// Exposure by sport
    pub by_sport: HashMap<String, Decimal>,
    /// Live betting exposure
    pub live_betting: Decimal,
    /// Parlay exposure
    pub parlays: Decimal,
    /// Open bets
    pub open_bets: Vec<OpenBet>,
}

impl Default for ExposureTracking {
    fn default() -> Self {
        Self {
            daily: Decimal::ZERO,
            weekly: Decimal::ZERO,
            by_sport: HashMap::new(),
            live_betting: Decimal::ZERO,
            parlays: Decimal::ZERO,
            open_bets: Vec::new(),
        }
    }
}

/// Open bet tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenBet {
    /// Bet ID
    pub bet_id: String,
    /// Sport
    pub sport: String,
    /// Amount
    pub amount: Decimal,
    /// Placement timestamp
    pub placed_at: DateTime<Utc>,
}

/// Withdrawal request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithdrawalRequest {
    /// Request ID
    pub id: Uuid,
    /// Member ID
    pub member_id: Uuid,
    /// Requested amount
    pub requested_amount: Decimal,
    /// Approved amount
    pub approved_amount: Decimal,
    /// Penalty amount
    pub penalty: Decimal,
    /// Net amount after fees
    pub net_amount: Decimal,
    /// Is emergency withdrawal
    pub is_emergency: bool,
    /// Request status
    pub status: String,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Scheduled processing time
    pub scheduled_for: DateTime<Utc>,
}

/// Vote proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteProposal {
    /// Vote ID
    pub id: Uuid,
    /// Proposal type
    pub proposal_type: String,
    /// Proposal details (JSON)
    pub details: String,
    /// Proposed by member ID
    pub proposed_by: Uuid,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Expiration timestamp
    pub expires_at: DateTime<Utc>,
    /// Vote status
    pub status: String,
}

/// Individual vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vote {
    /// Member ID
    pub member_id: Uuid,
    /// Decision (approve/reject/abstain)
    pub decision: String,
    /// Voting weight
    pub weight: f64,
    /// Vote timestamp
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_bankroll_rules() {
        let rules = BankrollRules::default();
        assert_eq!(rules.max_single_bet, 0.05);
        assert_eq!(rules.max_daily_exposure, 0.20);
    }

    #[test]
    fn test_member_permissions_for_role() {
        let lead_perms = MemberPermissions::for_role(MemberRole::LeadInvestor);
        assert!(lead_perms.create_syndicate);
        assert!(lead_perms.veto_power);

        let observer_perms = MemberPermissions::for_role(MemberRole::Observer);
        assert!(!observer_perms.propose_bets);
        assert!(observer_perms.view_bets);
    }

    #[test]
    fn test_allocation_strategy() {
        let strategy = AllocationStrategy::KellyCriterion;
        assert_eq!(strategy, AllocationStrategy::KellyCriterion);
    }
}
