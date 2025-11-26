//! Member management and performance tracking

use crate::types::*;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use napi_derive::napi;
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// Syndicate member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Member {
    /// Member ID
    pub id: Uuid,
    /// Member name
    pub name: String,
    /// Email address
    pub email: String,
    /// Member role
    pub role: MemberRole,
    /// Member tier
    pub tier: MemberTier,
    /// Member permissions
    pub permissions: MemberPermissions,
    /// Capital contribution
    pub capital_contribution: Decimal,
    /// Performance score
    pub performance_score: f64,
    /// ROI score
    pub roi_score: f64,
    /// Accuracy score
    pub accuracy_score: f64,
    /// Statistics
    pub statistics: MemberStatistics,
    /// Is active
    pub is_active: bool,
    /// Join date
    pub joined_date: DateTime<Utc>,
}

impl Member {
    /// Create new member
    pub fn new(name: String, email: String, role: MemberRole) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            email,
            role: role.clone(),
            tier: MemberTier::Bronze,
            permissions: MemberPermissions::for_role(role),
            capital_contribution: Decimal::ZERO,
            performance_score: 0.0,
            roi_score: 0.0,
            accuracy_score: 0.0,
            statistics: MemberStatistics::default(),
            is_active: true,
            joined_date: Utc::now(),
        }
    }

    /// Update member tier based on contribution
    pub fn update_tier(&mut self, contribution: Decimal) {
        self.capital_contribution = contribution;

        self.tier = if contribution >= Decimal::from(100000) {
            MemberTier::Platinum
        } else if contribution >= Decimal::from(25000) {
            MemberTier::Gold
        } else if contribution >= Decimal::from(5000) {
            MemberTier::Silver
        } else {
            MemberTier::Bronze
        };
    }

    /// Calculate voting weight
    pub fn calculate_voting_weight(&self, syndicate_total_capital: Decimal) -> f64 {
        if !self.is_active || syndicate_total_capital == Decimal::ZERO {
            return 0.0;
        }

        // Capital weight (50%)
        let capital_weight = (self.capital_contribution / syndicate_total_capital).to_f64().unwrap_or(0.0) * 0.5;

        // Performance weight (30%)
        let performance_weight = self.performance_score * 0.3;

        // Tenure weight (20%)
        let months_active = (Utc::now() - self.joined_date).num_days() as f64 / 30.0;
        let tenure_weight = (months_active / 12.0).min(1.0) * 0.2;

        // Role multiplier
        let role_multiplier = match self.role {
            MemberRole::LeadInvestor => 1.5,
            MemberRole::SeniorAnalyst => 1.3,
            MemberRole::JuniorAnalyst => 1.1,
            MemberRole::ContributingMember => 1.0,
            MemberRole::Observer => 0.0,
        };

        let base_weight = capital_weight + performance_weight + tenure_weight;
        base_weight * role_multiplier
    }
}

/// Member manager
#[napi]
#[derive(Clone)]
pub struct MemberManager {
    syndicate_id: String,
    members: Arc<DashMap<String, Member>>,
    max_members: usize,
}

#[napi]
impl MemberManager {
    /// Create new member manager
    #[napi(constructor)]
    pub fn new(syndicate_id: String) -> Self {
        Self {
            syndicate_id,
            members: Arc::new(DashMap::new()),
            max_members: 100,
        }
    }

    /// Add new member
    #[napi]
    pub fn add_member(
        &self,
        name: String,
        email: String,
        role: MemberRole,
        initial_contribution: String,
    ) -> napi::Result<String> {
        if self.members.len() >= self.max_members {
            return Err(napi::Error::from_reason("Maximum members reached"));
        }

        let contribution = Decimal::from_str(&initial_contribution)
            .map_err(|e| napi::Error::from_reason(format!("Invalid contribution: {}", e)))?;

        let mut member = Member::new(name, email, role);
        member.update_tier(contribution);

        let member_id = member.id.to_string();
        self.members.insert(member_id.clone(), member.clone());

        serde_json::to_string(&member)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Update member role
    #[napi]
    pub fn update_member_role(
        &self,
        member_id: String,
        new_role: MemberRole,
        authorized_by: String,
    ) -> napi::Result<()> {
        // Check authorization
        let authorizer = self.members.get(&authorized_by)
            .ok_or_else(|| napi::Error::from_reason("Authorizer not found"))?;

        if !authorizer.permissions.manage_members {
            return Err(napi::Error::from_reason("Not authorized to manage members"));
        }

        // Update member
        let mut member = self.members.get_mut(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        member.role = new_role.clone();
        member.permissions = MemberPermissions::for_role(new_role);

        Ok(())
    }

    /// Suspend member
    #[napi]
    pub fn suspend_member(
        &self,
        member_id: String,
        _reason: String,
        authorized_by: String,
    ) -> napi::Result<()> {
        let authorizer = self.members.get(&authorized_by)
            .ok_or_else(|| napi::Error::from_reason("Authorizer not found"))?;

        if !authorizer.permissions.manage_members {
            return Err(napi::Error::from_reason("Not authorized"));
        }

        let mut member = self.members.get_mut(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        member.is_active = false;

        Ok(())
    }

    /// Update member capital contribution
    #[napi]
    pub fn update_contribution(&self, member_id: String, amount: String) -> napi::Result<()> {
        let contribution = Decimal::from_str(&amount)
            .map_err(|e| napi::Error::from_reason(format!("Invalid amount: {}", e)))?;

        let mut member = self.members.get_mut(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        member.update_tier(contribution);

        Ok(())
    }

    /// Track bet outcome for member performance
    #[napi]
    pub fn track_bet_outcome(&self, member_id: String, bet_details: String) -> napi::Result<()> {
        let bet: serde_json::Value = serde_json::from_str(&bet_details)
            .map_err(|e| napi::Error::from_reason(format!("Invalid bet data: {}", e)))?;

        let mut member = self.members.get_mut(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        // Update statistics
        member.statistics.bets_proposed += 1;

        let outcome = bet["outcome"].as_str().unwrap_or("");
        if outcome == "won" {
            member.statistics.bets_won += 1;
        } else if outcome == "lost" {
            member.statistics.bets_lost += 1;
        }

        // Update profit
        if let Some(profit_str) = bet["profit"].as_str() {
            if let Ok(profit) = Decimal::from_str(profit_str) {
                let current_profit = Decimal::from_str(&member.statistics.total_profit).unwrap_or(Decimal::ZERO);
                member.statistics.total_profit = (current_profit + profit).to_string();
            }
        }

        // Update staked amount
        if let Some(stake_str) = bet["stake"].as_str() {
            if let Ok(stake) = Decimal::from_str(stake_str) {
                let current_staked = Decimal::from_str(&member.statistics.total_staked).unwrap_or(Decimal::ZERO);
                member.statistics.total_staked = (current_staked + stake).to_string();
            }
        }

        // Recalculate win rate and ROI
        let total_bets = member.statistics.bets_won + member.statistics.bets_lost;
        if total_bets > 0 {
            member.statistics.win_rate = member.statistics.bets_won as f64 / total_bets as f64;
        }

        let total_staked = Decimal::from_str(&member.statistics.total_staked).unwrap_or(Decimal::ZERO);
        let total_profit = Decimal::from_str(&member.statistics.total_profit).unwrap_or(Decimal::ZERO);
        if total_staked > Decimal::ZERO {
            member.statistics.roi = (total_profit / total_staked * Decimal::from(100))
                .to_f64()
                .unwrap_or(0.0);
        }

        Ok(())
    }

    /// Get member performance report
    #[napi]
    pub fn get_member_performance_report(&self, member_id: String) -> napi::Result<String> {
        let member = self.members.get(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        let total_capital = self.get_total_capital_decimal();

        let report = serde_json::json!({
            "member_info": {
                "id": member.id.to_string(),
                "name": member.name,
                "role": format!("{:?}", member.role),
                "tier": format!("{:?}", member.tier),
                "joined_date": member.joined_date.to_rfc3339(),
                "is_active": member.is_active,
            },
            "financial_summary": {
                "capital_contribution": member.capital_contribution.to_string(),
                "total_profit": member.statistics.total_profit,
                "roi": member.statistics.roi,
            },
            "betting_performance": {
                "total_bets": member.statistics.bets_proposed,
                "bets_won": member.statistics.bets_won,
                "bets_lost": member.statistics.bets_lost,
                "win_rate": member.statistics.win_rate,
            },
            "voting_weight": member.calculate_voting_weight(total_capital),
        });

        serde_json::to_string(&report)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Get total syndicate capital
    #[napi]
    pub fn get_total_capital(&self) -> String {
        let total: Decimal = self.members
            .iter()
            .filter(|entry| entry.value().is_active)
            .map(|entry| entry.value().capital_contribution)
            .sum();
        total.to_string()
    }

    fn get_total_capital_decimal(&self) -> Decimal {
        self.members
            .iter()
            .filter(|entry| entry.value().is_active)
            .map(|entry| entry.value().capital_contribution)
            .sum()
    }

    /// List all members
    #[napi]
    pub fn list_members(&self, active_only: bool) -> napi::Result<String> {
        let members: Vec<_> = self.members
            .iter()
            .filter(|entry| !active_only || entry.value().is_active)
            .map(|entry| entry.value().clone())
            .collect();

        serde_json::to_string(&members)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Get member by ID
    #[napi]
    pub fn get_member(&self, member_id: String) -> napi::Result<String> {
        let member = self.members.get(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        serde_json::to_string(&*member)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Update member statistics
    #[napi]
    pub fn update_member_statistics(
        &self,
        member_id: String,
        statistics: String,
    ) -> napi::Result<()> {
        let stats: MemberStatistics = serde_json::from_str(&statistics)
            .map_err(|e| napi::Error::from_reason(format!("Invalid statistics: {}", e)))?;

        let mut member = self.members.get_mut(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        member.statistics = stats;

        Ok(())
    }

    /// Calculate member alpha (skill-based returns)
    #[napi]
    pub fn calculate_member_alpha(&self, member_id: String) -> napi::Result<String> {
        let member = self.members.get(&member_id)
            .ok_or_else(|| napi::Error::from_reason("Member not found"))?;

        let total_bets = member.statistics.bets_won + member.statistics.bets_lost;

        if total_bets < 10 {
            return Ok(serde_json::json!({
                "alpha": 0.0,
                "confidence": 0.1,
                "sample_size": total_bets,
                "message": "Insufficient sample size",
            }).to_string());
        }

        // Calculate actual ROI
        let actual_roi = member.statistics.roi / 100.0;

        // Assume market efficiency (expected negative return)
        let expected_roi = -0.05; // -5% expected return

        // Alpha is excess return
        let alpha = actual_roi - expected_roi;

        // Confidence based on sample size
        let sample_confidence = (total_bets as f64 / 100.0).min(1.0);

        // Consistency score (simplified)
        let consistency_score = if member.statistics.win_rate > 0.4 && member.statistics.win_rate < 0.6 {
            0.8 // More consistent around 50%
        } else {
            0.5
        };

        let confidence = sample_confidence * consistency_score;

        let result = serde_json::json!({
            "alpha": alpha,
            "confidence": confidence,
            "sample_size": total_bets,
            "consistency_score": consistency_score,
            "actual_roi": actual_roi,
            "expected_roi": expected_roi,
        });

        Ok(result.to_string())
    }

    /// Get member count
    #[napi]
    pub fn get_member_count(&self) -> u32 {
        self.members.len() as u32
    }

    /// Get active member count
    #[napi]
    pub fn get_active_member_count(&self) -> u32 {
        self.members
            .iter()
            .filter(|entry| entry.value().is_active)
            .count() as u32
    }
}

/// Performance tracker
#[napi]
#[derive(Clone)]
pub struct MemberPerformanceTracker {
    performance_history: Arc<DashMap<String, Vec<PerformanceRecord>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceRecord {
    timestamp: DateTime<Utc>,
    bet_id: String,
    sport: String,
    bet_type: String,
    odds: f64,
    stake: Decimal,
    outcome: String,
    profit: Decimal,
    confidence: f64,
    edge: f64,
}

#[napi]
impl MemberPerformanceTracker {
    /// Create new performance tracker
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            performance_history: Arc::new(DashMap::new()),
        }
    }

    /// Track bet outcome
    #[napi]
    pub fn track_bet_outcome(&self, member_id: String, bet_details: String) -> napi::Result<()> {
        let bet: serde_json::Value = serde_json::from_str(&bet_details)
            .map_err(|e| napi::Error::from_reason(format!("Invalid bet data: {}", e)))?;

        let record = PerformanceRecord {
            timestamp: Utc::now(),
            bet_id: bet["bet_id"].as_str().unwrap_or("").to_string(),
            sport: bet["sport"].as_str().unwrap_or("").to_string(),
            bet_type: bet["bet_type"].as_str().unwrap_or("").to_string(),
            odds: bet["odds"].as_f64().unwrap_or(0.0),
            stake: Decimal::from_str(bet["stake"].as_str().unwrap_or("0")).unwrap_or(Decimal::ZERO),
            outcome: bet["outcome"].as_str().unwrap_or("").to_string(),
            profit: Decimal::from_str(bet["profit"].as_str().unwrap_or("0")).unwrap_or(Decimal::ZERO),
            confidence: bet["confidence"].as_f64().unwrap_or(0.5),
            edge: bet["edge"].as_f64().unwrap_or(0.0),
        };

        self.performance_history
            .entry(member_id)
            .or_insert_with(Vec::new)
            .push(record);

        Ok(())
    }

    /// Get performance history
    #[napi]
    pub fn get_performance_history(&self, member_id: String) -> napi::Result<String> {
        let history = self.performance_history
            .get(&member_id)
            .map(|h| h.clone())
            .unwrap_or_default();

        serde_json::to_string(&history)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Identify member strengths
    #[napi]
    pub fn identify_member_strengths(&self, member_id: String) -> napi::Result<String> {
        let history = self.performance_history
            .get(&member_id)
            .map(|h| h.clone())
            .unwrap_or_default();

        if history.len() < 10 {
            return Ok(serde_json::json!({
                "strengths": [],
                "weaknesses": [],
                "recommendations": ["Need more betting history for analysis"],
            }).to_string());
        }

        let mut sport_stats: HashMap<String, (u32, u32, Decimal, Decimal)> = HashMap::new();

        for record in &history {
            let entry = sport_stats.entry(record.sport.clone()).or_insert((0, 0, Decimal::ZERO, Decimal::ZERO));
            entry.0 += 1; // Total bets
            if record.outcome == "won" {
                entry.1 += 1; // Wins
            }
            entry.2 += record.stake; // Total staked
            entry.3 += record.profit; // Total profit
        }

        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();

        for (sport, (total, wins, staked, profit)) in sport_stats {
            if total >= 5 {
                let win_rate = wins as f64 / total as f64;
                let roi = if staked > Decimal::ZERO {
                    (profit / staked).to_f64().unwrap_or(0.0)
                } else {
                    0.0
                };

                if win_rate > 0.55 && roi > 0.05 {
                    strengths.push(format!("Strong performance in {} (WR: {:.1}%, ROI: {:.1}%)",
                        sport, win_rate * 100.0, roi * 100.0));
                } else if win_rate < 0.45 || roi < -0.10 {
                    weaknesses.push(format!("Underperforming in {} (WR: {:.1}%, ROI: {:.1}%)",
                        sport, win_rate * 100.0, roi * 100.0));
                }
            }
        }

        let recommendations = if strengths.is_empty() {
            vec!["Focus on improving analytical skills and bet selection".to_string()]
        } else {
            vec![format!("Continue focusing on your strengths: {}", strengths.join(", "))]
        };

        let result = serde_json::json!({
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendations": recommendations,
        });

        Ok(result.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_member_creation() {
        let member = Member::new("John Doe".to_string(), "john@example.com".to_string(), MemberRole::ContributingMember);
        assert_eq!(member.name, "John Doe");
        assert_eq!(member.role, MemberRole::ContributingMember);
        assert!(member.is_active);
    }

    #[test]
    fn test_member_tier_update() {
        let mut member = Member::new("Jane".to_string(), "jane@example.com".to_string(), MemberRole::ContributingMember);
        member.update_tier(Decimal::from(50000));
        assert_eq!(member.tier, MemberTier::Gold);
    }

    #[test]
    fn test_member_manager() {
        let manager = MemberManager::new("test-syndicate".to_string());
        assert_eq!(manager.get_member_count(), 0);
    }
}
