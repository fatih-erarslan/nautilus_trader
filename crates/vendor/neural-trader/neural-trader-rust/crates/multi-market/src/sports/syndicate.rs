//! Syndicate Management for Sports Betting
//!
//! Manages pooled betting syndicates with multiple members

use crate::error::{MultiMarketError, Result};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Member role in syndicate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemberRole {
    /// Syndicate manager (full control)
    Manager,
    /// Active betting member
    Member,
    /// Observer only (no betting)
    Observer,
}

/// Syndicate member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    /// Member identifier
    pub id: String,
    /// Member name
    pub name: String,
    /// Member email
    pub email: String,
    /// Member role
    pub role: MemberRole,
    /// Capital contributed
    pub capital_contributed: Decimal,
    /// Current share percentage
    pub share_percentage: Decimal,
    /// Total profit/loss
    pub total_pnl: Decimal,
    /// Join date
    pub joined_at: DateTime<Utc>,
    /// Active status
    pub active: bool,
}

/// Bet placed by syndicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndicateBet {
    /// Bet identifier
    pub id: String,
    /// Event identifier
    pub event_id: String,
    /// Event description
    pub event_description: String,
    /// Bookmaker
    pub bookmaker: String,
    /// Outcome bet on
    pub outcome: String,
    /// Odds
    pub odds: Decimal,
    /// Stake amount
    pub stake: Decimal,
    /// Potential payout
    pub potential_payout: Decimal,
    /// Actual payout (if settled)
    pub actual_payout: Option<Decimal>,
    /// Bet status
    pub status: BetStatus,
    /// Placed at
    pub placed_at: DateTime<Utc>,
    /// Settled at
    pub settled_at: Option<DateTime<Utc>>,
}

/// Bet status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BetStatus {
    /// Bet placed, awaiting settlement
    Pending,
    /// Bet won
    Won,
    /// Bet lost
    Lost,
    /// Bet voided/cancelled
    Void,
}

/// Profit distribution event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitDistribution {
    /// Distribution identifier
    pub id: String,
    /// Total profit distributed
    pub total_profit: Decimal,
    /// Distribution to each member
    pub member_distributions: HashMap<String, Decimal>,
    /// Distribution date
    pub distributed_at: DateTime<Utc>,
}

/// Withdrawal request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithdrawalRequest {
    /// Request identifier
    pub id: String,
    /// Member ID
    pub member_id: String,
    /// Withdrawal amount
    pub amount: Decimal,
    /// Request status
    pub status: WithdrawalStatus,
    /// Request date
    pub requested_at: DateTime<Utc>,
    /// Processing date
    pub processed_at: Option<DateTime<Utc>>,
}

/// Withdrawal status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WithdrawalStatus {
    /// Request pending approval
    Pending,
    /// Request approved
    Approved,
    /// Request rejected
    Rejected,
    /// Withdrawal completed
    Completed,
}

/// Betting syndicate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Syndicate {
    /// Syndicate identifier
    pub id: String,
    /// Syndicate name
    pub name: String,
    /// Syndicate description
    pub description: String,
    /// Total pooled capital
    pub total_capital: Decimal,
    /// Available capital for betting
    pub available_capital: Decimal,
    /// Members
    pub members: HashMap<String, Member>,
    /// Betting history
    pub bets: Vec<SyndicateBet>,
    /// Total profit/loss
    pub total_pnl: Decimal,
    /// Profit distributions
    pub distributions: Vec<ProfitDistribution>,
    /// Withdrawal requests
    pub withdrawal_requests: Vec<WithdrawalRequest>,
    /// Created at
    pub created_at: DateTime<Utc>,
    /// Updated at
    pub updated_at: DateTime<Utc>,
}

impl Syndicate {
    /// Create new syndicate
    pub fn new(id: String, name: String, description: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            description,
            total_capital: Decimal::ZERO,
            available_capital: Decimal::ZERO,
            members: HashMap::new(),
            bets: Vec::new(),
            total_pnl: Decimal::ZERO,
            distributions: Vec::new(),
            withdrawal_requests: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add member to syndicate
    pub fn add_member(
        &mut self,
        id: String,
        name: String,
        email: String,
        role: MemberRole,
        capital: Decimal,
    ) -> Result<()> {
        if self.members.contains_key(&id) {
            return Err(MultiMarketError::ValidationError(
                "Member already exists".to_string(),
            ));
        }

        if capital <= Decimal::ZERO {
            return Err(MultiMarketError::ValidationError(
                "Capital must be positive".to_string(),
            ));
        }

        let member = Member {
            id: id.clone(),
            name,
            email,
            role,
            capital_contributed: capital,
            share_percentage: Decimal::ZERO, // Will be recalculated
            total_pnl: Decimal::ZERO,
            joined_at: Utc::now(),
            active: true,
        };

        self.members.insert(id, member);
        self.total_capital += capital;
        self.available_capital += capital;
        self.recalculate_shares();
        self.updated_at = Utc::now();

        Ok(())
    }

    /// Remove member from syndicate
    pub fn remove_member(&mut self, member_id: &str) -> Result<()> {
        let member = self
            .members
            .get_mut(member_id)
            .ok_or_else(|| MultiMarketError::ValidationError("Member not found".to_string()))?;

        member.active = false;
        self.updated_at = Utc::now();

        Ok(())
    }

    /// Place bet for syndicate
    pub fn place_bet(
        &mut self,
        bet_id: String,
        event_id: String,
        event_description: String,
        bookmaker: String,
        outcome: String,
        odds: Decimal,
        stake: Decimal,
    ) -> Result<()> {
        if stake > self.available_capital {
            return Err(MultiMarketError::ValidationError(
                "Insufficient available capital".to_string(),
            ));
        }

        if stake <= Decimal::ZERO {
            return Err(MultiMarketError::ValidationError(
                "Stake must be positive".to_string(),
            ));
        }

        let bet = SyndicateBet {
            id: bet_id,
            event_id,
            event_description,
            bookmaker,
            outcome,
            odds,
            stake,
            potential_payout: stake * odds,
            actual_payout: None,
            status: BetStatus::Pending,
            placed_at: Utc::now(),
            settled_at: None,
        };

        self.available_capital -= stake;
        self.bets.push(bet);
        self.updated_at = Utc::now();

        Ok(())
    }

    /// Settle bet
    pub fn settle_bet(&mut self, bet_id: &str, won: bool) -> Result<Decimal> {
        let bet = self
            .bets
            .iter_mut()
            .find(|b| b.id == bet_id)
            .ok_or_else(|| MultiMarketError::ValidationError("Bet not found".to_string()))?;

        if bet.status != BetStatus::Pending {
            return Err(MultiMarketError::ValidationError(
                "Bet already settled".to_string(),
            ));
        }

        let pnl = if won {
            bet.status = BetStatus::Won;
            let payout = bet.potential_payout;
            bet.actual_payout = Some(payout);
            self.available_capital += payout;
            payout - bet.stake
        } else {
            bet.status = BetStatus::Lost;
            bet.actual_payout = Some(Decimal::ZERO);
            -bet.stake
        };

        bet.settled_at = Some(Utc::now());
        self.total_pnl += pnl;
        self.updated_at = Utc::now();

        Ok(pnl)
    }

    /// Distribute profits to members
    pub fn distribute_profits(&mut self) -> Result<ProfitDistribution> {
        if self.total_pnl <= Decimal::ZERO {
            return Err(MultiMarketError::ValidationError(
                "No profits to distribute".to_string(),
            ));
        }

        let mut member_distributions = HashMap::new();

        for (member_id, member) in self.members.iter_mut() {
            if !member.active {
                continue;
            }

            let member_profit = self.total_pnl * member.share_percentage;
            member.total_pnl += member_profit;
            member_distributions.insert(member_id.clone(), member_profit);
        }

        let distribution = ProfitDistribution {
            id: uuid::Uuid::new_v4().to_string(),
            total_profit: self.total_pnl,
            member_distributions,
            distributed_at: Utc::now(),
        };

        self.distributions.push(distribution.clone());
        self.total_pnl = Decimal::ZERO; // Reset after distribution
        self.updated_at = Utc::now();

        Ok(distribution)
    }

    /// Request withdrawal
    pub fn request_withdrawal(
        &mut self,
        member_id: String,
        amount: Decimal,
    ) -> Result<WithdrawalRequest> {
        let member = self
            .members
            .get(&member_id)
            .ok_or_else(|| MultiMarketError::ValidationError("Member not found".to_string()))?;

        if !member.active {
            return Err(MultiMarketError::ValidationError(
                "Member not active".to_string(),
            ));
        }

        let member_capital = member.capital_contributed + member.total_pnl;
        if amount > member_capital {
            return Err(MultiMarketError::ValidationError(
                "Withdrawal exceeds member capital".to_string(),
            ));
        }

        let request = WithdrawalRequest {
            id: uuid::Uuid::new_v4().to_string(),
            member_id,
            amount,
            status: WithdrawalStatus::Pending,
            requested_at: Utc::now(),
            processed_at: None,
        };

        self.withdrawal_requests.push(request.clone());
        self.updated_at = Utc::now();

        Ok(request)
    }

    /// Process withdrawal request
    pub fn process_withdrawal(&mut self, request_id: &str, approve: bool) -> Result<()> {
        let request_index = self
            .withdrawal_requests
            .iter()
            .position(|r| r.id == request_id)
            .ok_or_else(|| MultiMarketError::ValidationError("Request not found".to_string()))?;

        let request = &mut self.withdrawal_requests[request_index];

        if request.status != WithdrawalStatus::Pending {
            return Err(MultiMarketError::ValidationError(
                "Request already processed".to_string(),
            ));
        }

        let amount = request.amount;
        let member_id = request.member_id.clone();

        if approve {
            if amount > self.available_capital {
                return Err(MultiMarketError::ValidationError(
                    "Insufficient available capital".to_string(),
                ));
            }

            request.status = WithdrawalStatus::Completed;
            request.processed_at = Some(Utc::now());

            self.available_capital -= amount;
            self.total_capital -= amount;

            // Update member capital
            if let Some(member) = self.members.get_mut(&member_id) {
                member.capital_contributed -= amount;
            }

            self.recalculate_shares();
        } else {
            request.status = WithdrawalStatus::Rejected;
            request.processed_at = Some(Utc::now());
        }

        self.updated_at = Utc::now();

        Ok(())
    }

    /// Get member by ID
    pub fn get_member(&self, member_id: &str) -> Option<&Member> {
        self.members.get(member_id)
    }

    /// Get active members
    pub fn active_members(&self) -> Vec<&Member> {
        self.members.values().filter(|m| m.active).collect()
    }

    /// Calculate win rate
    pub fn win_rate(&self) -> Decimal {
        let settled_bets: Vec<_> = self
            .bets
            .iter()
            .filter(|b| b.status == BetStatus::Won || b.status == BetStatus::Lost)
            .collect();

        if settled_bets.is_empty() {
            return Decimal::ZERO;
        }

        let won_bets = settled_bets.iter().filter(|b| b.status == BetStatus::Won).count();
        Decimal::from(won_bets) / Decimal::from(settled_bets.len())
    }

    /// Calculate ROI
    pub fn roi(&self) -> Decimal {
        if self.total_capital == Decimal::ZERO {
            return Decimal::ZERO;
        }

        (self.total_pnl / self.total_capital) * Decimal::from(100)
    }

    /// Recalculate member share percentages
    fn recalculate_shares(&mut self) {
        if self.total_capital == Decimal::ZERO {
            return;
        }

        for member in self.members.values_mut() {
            if member.active {
                member.share_percentage = member.capital_contributed / self.total_capital;
            } else {
                member.share_percentage = Decimal::ZERO;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syndicate_creation() {
        let syndicate = Syndicate::new(
            "test_syndicate".to_string(),
            "Test Syndicate".to_string(),
            "A test betting syndicate".to_string(),
        );

        assert_eq!(syndicate.id, "test_syndicate");
        assert_eq!(syndicate.total_capital, Decimal::ZERO);
        assert_eq!(syndicate.members.len(), 0);
    }

    #[test]
    fn test_add_member() {
        let mut syndicate = Syndicate::new(
            "test".to_string(),
            "Test".to_string(),
            "Test".to_string(),
        );

        syndicate
            .add_member(
                "member1".to_string(),
                "John Doe".to_string(),
                "john@example.com".to_string(),
                MemberRole::Member,
                dec!(1000),
            )
            .unwrap();

        assert_eq!(syndicate.members.len(), 1);
        assert_eq!(syndicate.total_capital, dec!(1000));
        assert_eq!(syndicate.available_capital, dec!(1000));

        let member = syndicate.get_member("member1").unwrap();
        assert_eq!(member.share_percentage, Decimal::ONE); // 100%
    }

    #[test]
    fn test_place_bet() {
        let mut syndicate = Syndicate::new(
            "test".to_string(),
            "Test".to_string(),
            "Test".to_string(),
        );

        syndicate
            .add_member(
                "member1".to_string(),
                "John".to_string(),
                "john@example.com".to_string(),
                MemberRole::Manager,
                dec!(1000),
            )
            .unwrap();

        syndicate
            .place_bet(
                "bet1".to_string(),
                "event1".to_string(),
                "Lakers vs Warriors".to_string(),
                "draftkings".to_string(),
                "Lakers Win".to_string(),
                dec!(2.5),
                dec!(100),
            )
            .unwrap();

        assert_eq!(syndicate.bets.len(), 1);
        assert_eq!(syndicate.available_capital, dec!(900));
    }

    #[test]
    fn test_settle_bet_won() {
        let mut syndicate = Syndicate::new(
            "test".to_string(),
            "Test".to_string(),
            "Test".to_string(),
        );

        syndicate
            .add_member(
                "member1".to_string(),
                "John".to_string(),
                "john@example.com".to_string(),
                MemberRole::Manager,
                dec!(1000),
            )
            .unwrap();

        syndicate
            .place_bet(
                "bet1".to_string(),
                "event1".to_string(),
                "Event".to_string(),
                "bookmaker".to_string(),
                "outcome".to_string(),
                dec!(2.0),
                dec!(100),
            )
            .unwrap();

        let pnl = syndicate.settle_bet("bet1", true).unwrap();

        assert_eq!(pnl, dec!(100)); // Won 100 (200 payout - 100 stake)
        assert_eq!(syndicate.available_capital, dec!(1100)); // 900 + 200 payout
        assert_eq!(syndicate.total_pnl, dec!(100));
    }

    #[test]
    fn test_distribute_profits() {
        let mut syndicate = Syndicate::new(
            "test".to_string(),
            "Test".to_string(),
            "Test".to_string(),
        );

        syndicate
            .add_member(
                "member1".to_string(),
                "John".to_string(),
                "john@example.com".to_string(),
                MemberRole::Member,
                dec!(600),
            )
            .unwrap();

        syndicate
            .add_member(
                "member2".to_string(),
                "Jane".to_string(),
                "jane@example.com".to_string(),
                MemberRole::Member,
                dec!(400),
            )
            .unwrap();

        syndicate.total_pnl = dec!(100);

        let distribution = syndicate.distribute_profits().unwrap();

        // Member1: 60% of 100 = 60
        // Member2: 40% of 100 = 40
        assert_eq!(
            *distribution.member_distributions.get("member1").unwrap(),
            dec!(60)
        );
        assert_eq!(
            *distribution.member_distributions.get("member2").unwrap(),
            dec!(40)
        );
    }

    #[test]
    fn test_withdrawal_request() {
        let mut syndicate = Syndicate::new(
            "test".to_string(),
            "Test".to_string(),
            "Test".to_string(),
        );

        syndicate
            .add_member(
                "member1".to_string(),
                "John".to_string(),
                "john@example.com".to_string(),
                MemberRole::Member,
                dec!(1000),
            )
            .unwrap();

        let request = syndicate
            .request_withdrawal("member1".to_string(), dec!(500))
            .unwrap();

        assert_eq!(request.status, WithdrawalStatus::Pending);
        assert_eq!(request.amount, dec!(500));

        syndicate.process_withdrawal(&request.id, true).unwrap();

        assert_eq!(syndicate.available_capital, dec!(500));
        assert_eq!(syndicate.total_capital, dec!(500));
    }
}
