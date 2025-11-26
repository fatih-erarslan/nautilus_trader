//! Capital management and fund allocation

use crate::types::*;
use chrono::{DateTime, Utc};
use napi_derive::napi;
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fund allocation engine for automated bankroll management
#[napi]
#[derive(Debug, Clone)]
pub struct FundAllocationEngine {
    syndicate_id: String,
    total_bankroll: Decimal,
    rules: BankrollRules,
    current_exposure: ExposureTracking,
    allocation_history: Vec<AllocationLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AllocationLog {
    timestamp: DateTime<Utc>,
    opportunity_sport: String,
    opportunity_event: String,
    amount: Decimal,
    percentage: f64,
    approval_required: bool,
    warnings: Vec<String>,
}

#[napi]
impl FundAllocationEngine {
    /// Create new fund allocation engine
    #[napi(constructor)]
    pub fn new(syndicate_id: String, total_bankroll: String) -> napi::Result<Self> {
        let bankroll = Decimal::from_str(&total_bankroll)
            .map_err(|e| napi::Error::from_reason(format!("Invalid bankroll: {}", e)))?;

        Ok(Self {
            syndicate_id,
            total_bankroll: bankroll,
            rules: BankrollRules::default(),
            current_exposure: ExposureTracking::default(),
            allocation_history: Vec::new(),
        })
    }

    /// Allocate funds for a betting opportunity
    #[napi]
    pub fn allocate_funds(
        &mut self,
        opportunity: BettingOpportunity,
        strategy: AllocationStrategy,
    ) -> napi::Result<AllocationResult> {
        // Calculate base allocation
        let base_allocation = self.calculate_base_allocation(&opportunity, strategy)?;

        // Apply constraints
        let constrained_allocation = self.apply_constraints(base_allocation, &opportunity)?;

        // Generate reasoning
        let reasoning = self.generate_reasoning(&opportunity, strategy.clone(), base_allocation, constrained_allocation);

        // Calculate risk metrics
        let risk_metrics = self.calculate_risk_metrics(constrained_allocation, &opportunity);

        // Check approval requirements
        let approval_required = self.needs_approval(constrained_allocation);

        // Generate warnings
        let warnings = self.generate_warnings(constrained_allocation, &opportunity);

        // Calculate stake sizing options
        let stake_sizing = self.calculate_stake_sizing_options(constrained_allocation, &opportunity);

        // Log allocation
        self.log_allocation(&opportunity, constrained_allocation, &warnings, approval_required);

        Ok(AllocationResult {
            amount: constrained_allocation.to_string(),
            percentage_of_bankroll: constrained_allocation
                .checked_div(self.total_bankroll)
                .unwrap_or(Decimal::ZERO)
                .to_f64()
                .unwrap_or(0.0),
            reasoning: serde_json::to_string(&reasoning).unwrap_or_default(),
            risk_metrics: serde_json::to_string(&risk_metrics).unwrap_or_default(),
            approval_required,
            warnings,
            recommended_stake_sizing: serde_json::to_string(&stake_sizing).unwrap_or_default(),
        })
    }

    /// Update exposure tracking after bet placement
    #[napi]
    pub fn update_exposure(&mut self, bet_placed: String) -> napi::Result<()> {
        let bet: serde_json::Value = serde_json::from_str(&bet_placed)
            .map_err(|e| napi::Error::from_reason(format!("Invalid bet data: {}", e)))?;

        let amount = Decimal::from_str(bet["amount"].as_str().unwrap_or("0"))
            .map_err(|e| napi::Error::from_reason(format!("Invalid amount: {}", e)))?;

        self.current_exposure.daily += amount;
        self.current_exposure.weekly += amount;

        let sport = bet["sport"].as_str().unwrap_or("unknown").to_string();
        *self.current_exposure.by_sport.entry(sport.clone()).or_insert(Decimal::ZERO) += amount;

        if bet["is_live"].as_bool().unwrap_or(false) {
            self.current_exposure.live_betting += amount;
        }

        if bet["is_parlay"].as_bool().unwrap_or(false) {
            self.current_exposure.parlays += amount;
        }

        self.current_exposure.open_bets.push(OpenBet {
            bet_id: bet["bet_id"].as_str().unwrap_or("").to_string(),
            sport,
            amount,
            placed_at: Utc::now(),
        });

        Ok(())
    }

    /// Get current exposure summary
    #[napi]
    pub fn get_exposure_summary(&self) -> String {
        serde_json::to_string(&self.current_exposure).unwrap_or_default()
    }
}

impl FundAllocationEngine {
    fn calculate_base_allocation(
        &self,
        opportunity: &BettingOpportunity,
        strategy: AllocationStrategy,
    ) -> napi::Result<Decimal> {
        match strategy {
            AllocationStrategy::KellyCriterion => self.kelly_allocation(opportunity),
            AllocationStrategy::FixedPercentage => self.fixed_allocation(opportunity),
            AllocationStrategy::DynamicConfidence => self.confidence_based_allocation(opportunity),
            AllocationStrategy::RiskParity => self.risk_parity_allocation(opportunity),
            AllocationStrategy::Martingale => self.martingale_allocation(opportunity),
            AllocationStrategy::AntiMartingale => self.anti_martingale_allocation(opportunity),
        }
    }

    fn kelly_allocation(&self, opportunity: &BettingOpportunity) -> napi::Result<Decimal> {
        if opportunity.edge <= 0.0 || opportunity.probability <= 0.0 {
            return Ok(Decimal::ZERO);
        }

        // Kelly percentage = (bp - q) / b
        let b = opportunity.odds - 1.0;
        let p = opportunity.probability;
        let q = 1.0 - p;

        let kelly_percentage = (b * p - q) / b;

        // Fractional Kelly (25% of full Kelly for safety)
        let conservative_kelly = kelly_percentage * 0.25;

        // Adjust for confidence and model agreement
        let confidence_adjustment = opportunity.confidence * opportunity.model_agreement;
        let adjusted_kelly = conservative_kelly * confidence_adjustment;

        let allocation = self.total_bankroll
            * Decimal::from_f64(adjusted_kelly.max(0.0))
                .ok_or_else(|| napi::Error::from_reason("Invalid Kelly calculation"))?;

        Ok(allocation.round_dp(2))
    }

    fn fixed_allocation(&self, opportunity: &BettingOpportunity) -> napi::Result<Decimal> {
        let base_percentage = Decimal::from_str("0.02").unwrap(); // 2%

        let confidence_multiplier = Decimal::from_f64(opportunity.confidence)
            .ok_or_else(|| napi::Error::from_reason("Invalid confidence"))?;

        let edge_multiplier = Decimal::from_f64(1.0 + opportunity.edge)
            .ok_or_else(|| napi::Error::from_reason("Invalid edge"))?;

        let adjusted_percentage = base_percentage * confidence_multiplier * edge_multiplier;
        let allocation = self.total_bankroll * adjusted_percentage;

        Ok(allocation.round_dp(2))
    }

    fn confidence_based_allocation(&self, opportunity: &BettingOpportunity) -> napi::Result<Decimal> {
        let allocation_percentage = if opportunity.confidence >= 0.9 {
            Decimal::from_str("0.05").unwrap() // 5%
        } else if opportunity.confidence >= 0.8 {
            Decimal::from_str("0.04").unwrap() // 4%
        } else if opportunity.confidence >= 0.7 {
            Decimal::from_str("0.03").unwrap() // 3%
        } else if opportunity.confidence >= 0.6 {
            Decimal::from_str("0.02").unwrap() // 2%
        } else if opportunity.confidence >= 0.5 {
            Decimal::from_str("0.01").unwrap() // 1%
        } else {
            Decimal::from_str("0.005").unwrap() // 0.5%
        };

        let mut final_percentage = allocation_percentage;

        // Adjust for edge
        if opportunity.edge > 0.1 {
            final_percentage *= Decimal::from_str("1.5").unwrap();
        } else if opportunity.edge > 0.05 {
            final_percentage *= Decimal::from_str("1.25").unwrap();
        }

        let allocation = self.total_bankroll * final_percentage;
        Ok(allocation.round_dp(2))
    }

    fn risk_parity_allocation(&self, opportunity: &BettingOpportunity) -> napi::Result<Decimal> {
        let target_risk = Decimal::from_str("0.01").unwrap(); // 1% risk contribution

        // Estimate bet volatility
        let bet_volatility = Decimal::from_f64(1.0 / opportunity.odds.sqrt())
            .ok_or_else(|| napi::Error::from_reason("Invalid volatility calculation"))?;

        let allocation = (target_risk * self.total_bankroll)
            .checked_div(bet_volatility)
            .unwrap_or(Decimal::ZERO);

        // Adjust for correlation
        let correlation_adjustment = self.calculate_correlation_adjustment(opportunity);
        let final_allocation = allocation * correlation_adjustment;

        Ok(final_allocation.round_dp(2))
    }

    fn martingale_allocation(&self, opportunity: &BettingOpportunity) -> napi::Result<Decimal> {
        // Simple martingale: double after loss
        let base = self.total_bankroll * Decimal::from_str("0.01").unwrap(); // 1% base

        // Check for recent losses in allocation history
        let recent_losses = self.allocation_history.iter()
            .rev()
            .take(5)
            .filter(|log| log.opportunity_sport == opportunity.sport)
            .count();

        let multiplier = Decimal::from_u32(2_u32.pow(recent_losses as u32))
            .unwrap_or(Decimal::ONE);

        let allocation = base * multiplier;

        // Cap at max single bet
        let max_single = self.total_bankroll * Decimal::from_f64(self.rules.max_single_bet).unwrap();
        Ok(allocation.min(max_single).round_dp(2))
    }

    fn anti_martingale_allocation(&self, opportunity: &BettingOpportunity) -> napi::Result<Decimal> {
        // Anti-martingale: increase after wins
        let base = self.total_bankroll * Decimal::from_str("0.01").unwrap(); // 1% base

        // Check for recent wins
        let recent_wins = self.allocation_history.iter()
            .rev()
            .take(5)
            .filter(|log| log.opportunity_sport == opportunity.sport)
            .count();

        let multiplier = if recent_wins == 0 {
            Decimal::ONE
        } else {
            let base = Decimal::from_str("1.5").unwrap();
            let mut result = Decimal::ONE;
            for _ in 0..recent_wins {
                result *= base;
            }
            result
        };

        let allocation = base * multiplier;

        let max_single = self.total_bankroll * Decimal::from_f64(self.rules.max_single_bet).unwrap();
        Ok(allocation.min(max_single).round_dp(2))
    }

    fn apply_constraints(
        &self,
        base_allocation: Decimal,
        opportunity: &BettingOpportunity,
    ) -> napi::Result<Decimal> {
        let mut allocation = base_allocation;

        // Maximum single bet constraint
        let max_single = if opportunity.is_parlay {
            self.total_bankroll * Decimal::from_f64(self.rules.max_parlay_percentage).unwrap()
        } else {
            self.total_bankroll * Decimal::from_f64(self.rules.max_single_bet).unwrap()
        };
        allocation = allocation.min(max_single);

        // Daily exposure constraint
        let remaining_daily = (self.total_bankroll * Decimal::from_f64(self.rules.max_daily_exposure).unwrap())
            - self.current_exposure.daily;
        allocation = allocation.min(remaining_daily.max(Decimal::ZERO));

        // Sport concentration constraint
        let sport_exposure = self.current_exposure.by_sport.get(&opportunity.sport).unwrap_or(&Decimal::ZERO);
        let max_sport = self.total_bankroll * Decimal::from_f64(self.rules.max_sport_concentration).unwrap();
        let remaining_sport = max_sport - sport_exposure;
        allocation = allocation.min(remaining_sport.max(Decimal::ZERO));

        // Minimum reserve constraint
        let total_exposure = self.calculate_total_exposure();
        let available_funds = self.total_bankroll - total_exposure;
        let min_reserve = self.total_bankroll * Decimal::from_f64(self.rules.minimum_reserve).unwrap();
        let max_available = available_funds - min_reserve;
        allocation = allocation.min(max_available.max(Decimal::ZERO));

        // Live betting constraint
        if opportunity.is_live {
            let remaining_live = (self.total_bankroll * Decimal::from_f64(self.rules.max_live_betting).unwrap())
                - self.current_exposure.live_betting;
            allocation = allocation.min(remaining_live.max(Decimal::ZERO));
        }

        // Stop loss check
        if self.check_stop_loss() {
            allocation = Decimal::ZERO;
        }

        Ok(allocation.round_dp(2))
    }

    fn calculate_correlation_adjustment(&self, opportunity: &BettingOpportunity) -> Decimal {
        let same_sport_bets = self.current_exposure.open_bets.iter()
            .filter(|bet| bet.sport == opportunity.sport)
            .count();

        if same_sport_bets == 0 {
            Decimal::ONE
        } else {
            Decimal::ONE / (Decimal::ONE + Decimal::from_usize(same_sport_bets).unwrap() * Decimal::from_str("0.2").unwrap())
        }
    }

    fn calculate_total_exposure(&self) -> Decimal {
        self.current_exposure.open_bets.iter()
            .map(|bet| bet.amount)
            .sum()
    }

    fn check_stop_loss(&self) -> bool {
        // Placeholder - would check actual P&L
        false
    }

    fn needs_approval(&self, allocation: Decimal) -> bool {
        // Large bets need approval
        if allocation > self.total_bankroll * Decimal::from_str("0.05").unwrap() {
            return true;
        }

        // High daily exposure needs approval
        if (self.current_exposure.daily + allocation) > self.total_bankroll * Decimal::from_str("0.15").unwrap() {
            return true;
        }

        false
    }

    fn generate_warnings(&self, allocation: Decimal, opportunity: &BettingOpportunity) -> Vec<String> {
        let mut warnings = Vec::new();

        let percentage = allocation.checked_div(self.total_bankroll).unwrap_or(Decimal::ZERO).to_f64().unwrap_or(0.0);
        if percentage > 0.04 {
            warnings.push(format!("Large bet size: {:.1}% of bankroll", percentage * 100.0));
        }

        let daily_percentage = (self.current_exposure.daily + allocation)
            .checked_div(self.total_bankroll)
            .unwrap_or(Decimal::ZERO)
            .to_f64()
            .unwrap_or(0.0);
        if daily_percentage > 0.15 {
            warnings.push(format!("High daily exposure: {:.1}%", daily_percentage * 100.0));
        }

        if opportunity.edge < 0.03 {
            warnings.push(format!("Low edge: {:.1}%", opportunity.edge * 100.0));
        }

        if opportunity.liquidity < 10000.0 {
            warnings.push(format!("Low liquidity: ${:.0}", opportunity.liquidity));
        }

        warnings
    }

    fn generate_reasoning(
        &self,
        opportunity: &BettingOpportunity,
        strategy: AllocationStrategy,
        base_allocation: Decimal,
        final_allocation: Decimal,
    ) -> HashMap<String, serde_json::Value> {
        let mut reasoning = HashMap::new();

        reasoning.insert("strategy_used".to_string(), serde_json::json!(format!("{:?}", strategy)));
        reasoning.insert("base_amount".to_string(), serde_json::json!(base_allocation.to_string()));
        reasoning.insert("final_amount".to_string(), serde_json::json!(final_allocation.to_string()));
        reasoning.insert("edge".to_string(), serde_json::json!(opportunity.edge));
        reasoning.insert("confidence".to_string(), serde_json::json!(opportunity.confidence));

        if base_allocation > Decimal::ZERO {
            let reduction = ((base_allocation - final_allocation) / base_allocation * Decimal::from(100))
                .to_f64()
                .unwrap_or(0.0);
            reasoning.insert("reduction_percentage".to_string(), serde_json::json!(reduction));
        }

        reasoning
    }

    fn calculate_risk_metrics(&self, allocation: Decimal, opportunity: &BettingOpportunity) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        let allocation_f64 = allocation.to_f64().unwrap_or(0.0);

        metrics.insert("expected_value".to_string(), allocation_f64 * opportunity.edge);
        metrics.insert("value_at_risk_95".to_string(), allocation_f64 * 0.95);

        if opportunity.edge > 0.0 {
            let kelly_fraction = (allocation_f64 / self.total_bankroll.to_f64().unwrap_or(1.0))
                / (opportunity.edge / (opportunity.odds - 1.0));
            metrics.insert("kelly_fraction".to_string(), kelly_fraction);
        }

        let potential_profit = allocation_f64 * (opportunity.odds - 1.0);
        let risk_reward = potential_profit / allocation_f64;
        metrics.insert("risk_reward_ratio".to_string(), risk_reward);

        metrics
    }

    fn calculate_stake_sizing_options(
        &self,
        base_allocation: Decimal,
        _opportunity: &BettingOpportunity,
    ) -> HashMap<String, String> {
        let mut options = HashMap::new();

        options.insert("recommended".to_string(), base_allocation.round_dp(2).to_string());
        options.insert("conservative".to_string(), (base_allocation * Decimal::from_str("0.5").unwrap()).round_dp(2).to_string());

        let aggressive = (base_allocation * Decimal::from_str("1.5").unwrap())
            .min(self.total_bankroll * Decimal::from_f64(self.rules.max_single_bet).unwrap());
        options.insert("aggressive".to_string(), aggressive.round_dp(2).to_string());

        let minimum = (base_allocation * Decimal::from_str("0.25").unwrap())
            .min(Decimal::from(10));
        options.insert("minimum".to_string(), minimum.round_dp(2).to_string());

        options
    }

    fn log_allocation(
        &mut self,
        opportunity: &BettingOpportunity,
        allocation: Decimal,
        warnings: &[String],
        approval_required: bool,
    ) {
        self.allocation_history.push(AllocationLog {
            timestamp: Utc::now(),
            opportunity_sport: opportunity.sport.clone(),
            opportunity_event: opportunity.event.clone(),
            amount: allocation,
            percentage: allocation.checked_div(self.total_bankroll).unwrap_or(Decimal::ZERO).to_f64().unwrap_or(0.0),
            approval_required,
            warnings: warnings.to_vec(),
        });
    }
}

/// Profit distribution system
#[napi]
#[derive(Debug, Clone)]
pub struct ProfitDistributionSystem {
    syndicate_id: String,
    distribution_history: Vec<DistributionLog>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DistributionLog {
    timestamp: DateTime<Utc>,
    total_profit: Decimal,
    operational_reserve: Decimal,
    distributions: HashMap<String, MemberDistribution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemberDistribution {
    gross_amount: Decimal,
    tax_withheld: Decimal,
    net_amount: Decimal,
}

#[napi]
impl ProfitDistributionSystem {
    /// Create new profit distribution system
    #[napi(constructor)]
    pub fn new(syndicate_id: String) -> Self {
        Self {
            syndicate_id,
            distribution_history: Vec::new(),
        }
    }

    /// Calculate profit distribution for members
    #[napi]
    pub fn calculate_distribution(
        &mut self,
        total_profit: String,
        members_json: String,
        model: DistributionModel,
    ) -> napi::Result<String> {
        let profit = Decimal::from_str(&total_profit)
            .map_err(|e| napi::Error::from_reason(format!("Invalid profit: {}", e)))?;

        let members: Vec<serde_json::Value> = serde_json::from_str(&members_json)
            .map_err(|e| napi::Error::from_reason(format!("Invalid members JSON: {}", e)))?;

        // Reserve operational costs (5%)
        let operational_reserve = profit * Decimal::from_str("0.05").unwrap();
        let distributable_profit = profit - operational_reserve;

        let distributions = match model {
            DistributionModel::Hybrid => self.hybrid_distribution(distributable_profit, &members)?,
            DistributionModel::Proportional => self.proportional_distribution(distributable_profit, &members)?,
            DistributionModel::PerformanceWeighted => self.performance_weighted_distribution(distributable_profit, &members)?,
            DistributionModel::Tiered => self.tiered_distribution(distributable_profit, &members)?,
        };

        serde_json::to_string(&distributions)
            .map_err(|e| napi::Error::from_reason(format!("Serialization error: {}", e)))
    }
}

impl ProfitDistributionSystem {
    fn hybrid_distribution(
        &self,
        profit: Decimal,
        members: &[serde_json::Value],
    ) -> napi::Result<HashMap<String, serde_json::Value>> {
        let mut distributions = HashMap::new();

        let active_members: Vec<&serde_json::Value> = members.iter()
            .filter(|m| m["is_active"].as_bool().unwrap_or(false))
            .collect();

        if active_members.is_empty() {
            return Ok(distributions);
        }

        let total_capital: Decimal = active_members.iter()
            .map(|m| Decimal::from_str(m["capital_contribution"].as_str().unwrap_or("0")).unwrap_or(Decimal::ZERO))
            .sum();

        let total_performance: f64 = active_members.iter()
            .map(|m| m["performance_score"].as_f64().unwrap_or(0.0))
            .sum();

        let capital_portion = profit * Decimal::from_str("0.50").unwrap();
        let performance_portion = profit * Decimal::from_str("0.30").unwrap();
        let equal_portion = profit * Decimal::from_str("0.20").unwrap();

        for member in &active_members {
            let member_id = member["id"].as_str().unwrap_or("");
            let member_capital = Decimal::from_str(member["capital_contribution"].as_str().unwrap_or("0"))
                .unwrap_or(Decimal::ZERO);
            let performance_score = member["performance_score"].as_f64().unwrap_or(0.0);

            let capital_share = if total_capital > Decimal::ZERO {
                (member_capital / total_capital) * capital_portion
            } else {
                Decimal::ZERO
            };

            let performance_share = if total_performance > 0.0 {
                Decimal::from_f64(performance_score / total_performance).unwrap() * performance_portion
            } else {
                performance_portion / Decimal::from_usize(active_members.len()).unwrap()
            };

            let equal_share = equal_portion / Decimal::from_usize(active_members.len()).unwrap();

            let total_share = (capital_share + performance_share + equal_share).round_dp(2);

            distributions.insert(
                member_id.to_string(),
                serde_json::json!({
                    "gross_amount": total_share.to_string(),
                    "net_amount": total_share.to_string(),
                }),
            );
        }

        Ok(distributions)
    }

    fn proportional_distribution(
        &self,
        profit: Decimal,
        members: &[serde_json::Value],
    ) -> napi::Result<HashMap<String, serde_json::Value>> {
        let mut distributions = HashMap::new();

        let active_members: Vec<&serde_json::Value> = members.iter()
            .filter(|m| m["is_active"].as_bool().unwrap_or(false))
            .collect();

        let total_capital: Decimal = active_members.iter()
            .map(|m| Decimal::from_str(m["capital_contribution"].as_str().unwrap_or("0")).unwrap_or(Decimal::ZERO))
            .sum();

        if total_capital == Decimal::ZERO {
            return Ok(distributions);
        }

        for member in active_members {
            let member_id = member["id"].as_str().unwrap_or("");
            let member_capital = Decimal::from_str(member["capital_contribution"].as_str().unwrap_or("0"))
                .unwrap_or(Decimal::ZERO);

            let share = ((member_capital / total_capital) * profit).round_dp(2);

            distributions.insert(
                member_id.to_string(),
                serde_json::json!({
                    "gross_amount": share.to_string(),
                    "net_amount": share.to_string(),
                }),
            );
        }

        Ok(distributions)
    }

    fn performance_weighted_distribution(
        &self,
        profit: Decimal,
        members: &[serde_json::Value],
    ) -> napi::Result<HashMap<String, serde_json::Value>> {
        let mut distributions = HashMap::new();

        let active_members: Vec<&serde_json::Value> = members.iter()
            .filter(|m| m["is_active"].as_bool().unwrap_or(false))
            .collect();

        let total_score: f64 = active_members.iter()
            .map(|m| {
                let roi = m["roi_score"].as_f64().unwrap_or(0.0) * 0.6;
                let win_rate = m.get("win_rate").and_then(|v| v.as_f64()).unwrap_or(0.0) * 0.3;
                let consistency = 0.5 * 0.1;
                roi + win_rate + consistency
            })
            .sum();

        if total_score == 0.0 {
            return Ok(distributions);
        }

        for member in &active_members {
            let member_id = member["id"].as_str().unwrap_or("");
            let composite_score = {
                let roi = member["roi_score"].as_f64().unwrap_or(0.0) * 0.6;
                let win_rate = member.get("win_rate").and_then(|v| v.as_f64()).unwrap_or(0.0) * 0.3;
                let consistency = 0.5 * 0.1;
                roi + win_rate + consistency
            };

            let share = (Decimal::from_f64(composite_score / total_score).unwrap() * profit).round_dp(2);

            distributions.insert(
                member_id.to_string(),
                serde_json::json!({
                    "gross_amount": share.to_string(),
                    "net_amount": share.to_string(),
                }),
            );
        }

        Ok(distributions)
    }

    fn tiered_distribution(
        &self,
        profit: Decimal,
        members: &[serde_json::Value],
    ) -> napi::Result<HashMap<String, serde_json::Value>> {
        let mut distributions = HashMap::new();

        let active_members: Vec<&serde_json::Value> = members.iter()
            .filter(|m| m["is_active"].as_bool().unwrap_or(false))
            .collect();

        let total_weighted_units: f64 = active_members.iter()
            .map(|m| {
                match m["tier"].as_str().unwrap_or("bronze") {
                    "platinum" => 1.5,
                    "gold" => 1.2,
                    "silver" => 1.0,
                    "bronze" => 0.8,
                    _ => 1.0,
                }
            })
            .sum();

        for member in &active_members {
            let member_id = member["id"].as_str().unwrap_or("");
            let weight = match member["tier"].as_str().unwrap_or("bronze") {
                "platinum" => 1.5,
                "gold" => 1.2,
                "silver" => 1.0,
                "bronze" => 0.8,
                _ => 1.0,
            };

            let share = (Decimal::from_f64(weight / total_weighted_units).unwrap() * profit).round_dp(2);

            distributions.insert(
                member_id.to_string(),
                serde_json::json!({
                    "gross_amount": share.to_string(),
                    "net_amount": share.to_string(),
                }),
            );
        }

        Ok(distributions)
    }
}

/// Withdrawal manager
#[napi]
#[derive(Debug, Clone)]
pub struct WithdrawalManager {
    syndicate_id: String,
    withdrawal_requests: Vec<WithdrawalRequest>,
    rules: WithdrawalRules,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WithdrawalRules {
    minimum_notice_days: i64,
    maximum_withdrawal_percentage: f64,
    lockup_period_days: i64,
    emergency_withdrawal_penalty: f64,
}

impl Default for WithdrawalRules {
    fn default() -> Self {
        Self {
            minimum_notice_days: 7,
            maximum_withdrawal_percentage: 0.50,
            lockup_period_days: 90,
            emergency_withdrawal_penalty: 0.10,
        }
    }
}

#[napi]
impl WithdrawalManager {
    /// Create new withdrawal manager
    #[napi(constructor)]
    pub fn new(syndicate_id: String) -> Self {
        Self {
            syndicate_id,
            withdrawal_requests: Vec::new(),
            rules: WithdrawalRules::default(),
        }
    }

    /// Request withdrawal
    #[napi]
    pub fn request_withdrawal(
        &mut self,
        member_id: String,
        member_balance: String,
        amount: String,
        is_emergency: bool,
    ) -> napi::Result<String> {
        let balance = Decimal::from_str(&member_balance)
            .map_err(|e| napi::Error::from_reason(format!("Invalid balance: {}", e)))?;

        let requested_amount = Decimal::from_str(&amount)
            .map_err(|e| napi::Error::from_reason(format!("Invalid amount: {}", e)))?;

        let validation = self.validate_withdrawal(balance, requested_amount, is_emergency)?;

        if !validation["approved"].as_bool().unwrap_or(false) {
            return serde_json::to_string(&validation)
                .map_err(|e| napi::Error::from_reason(e.to_string()));
        }

        let approved_amount = Decimal::from_str(validation["approved_amount"].as_str().unwrap_or("0"))
            .unwrap_or(requested_amount);

        let (penalty, net_amount) = if is_emergency {
            let pen = approved_amount * Decimal::from_f64(self.rules.emergency_withdrawal_penalty).unwrap();
            (pen, approved_amount - pen)
        } else {
            (Decimal::ZERO, approved_amount)
        };

        let scheduled_date = if is_emergency {
            Utc::now() + chrono::Duration::days(1)
        } else {
            Utc::now() + chrono::Duration::days(self.rules.minimum_notice_days)
        };

        let request = WithdrawalRequest {
            id: uuid::Uuid::new_v4(),
            member_id: uuid::Uuid::parse_str(&member_id).unwrap_or_default(),
            requested_amount,
            approved_amount,
            penalty,
            net_amount,
            is_emergency,
            status: "scheduled".to_string(),
            requested_at: Utc::now(),
            scheduled_for: scheduled_date,
        };

        self.withdrawal_requests.push(request.clone());

        serde_json::to_string(&request)
            .map_err(|e| napi::Error::from_reason(e.to_string()))
    }

    /// Get withdrawal history
    #[napi]
    pub fn get_withdrawal_history(&self) -> String {
        serde_json::to_string(&self.withdrawal_requests).unwrap_or_default()
    }
}

impl WithdrawalManager {
    fn validate_withdrawal(
        &self,
        balance: Decimal,
        amount: Decimal,
        _is_emergency: bool,
    ) -> napi::Result<serde_json::Value> {
        let max_allowed = balance * Decimal::from_f64(self.rules.maximum_withdrawal_percentage).unwrap();

        if amount > max_allowed {
            return Ok(serde_json::json!({
                "approved": false,
                "reason": "Exceeds maximum withdrawal percentage",
                "approved_amount": max_allowed.to_string(),
            }));
        }

        let remaining = balance - amount;
        let min_balance = Decimal::from(100);

        if remaining < min_balance {
            let approved = balance - min_balance;
            return Ok(serde_json::json!({
                "approved": false,
                "reason": "Must maintain minimum balance",
                "approved_amount": approved.to_string(),
            }));
        }

        Ok(serde_json::json!({
            "approved": true,
            "approved_amount": amount.to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fund_allocation_engine() {
        let mut engine = FundAllocationEngine::new("test-syndicate".to_string(), "100000.00".to_string()).unwrap();
        assert_eq!(engine.total_bankroll, Decimal::from_str("100000.00").unwrap());
    }

    #[test]
    fn test_kelly_allocation() {
        let engine = FundAllocationEngine::new("test".to_string(), "10000.00".to_string()).unwrap();
        let opportunity = BettingOpportunity {
            sport: "football".to_string(),
            event: "Team A vs Team B".to_string(),
            bet_type: "moneyline".to_string(),
            selection: "Team A".to_string(),
            odds: 2.0,
            probability: 0.55,
            edge: 0.10,
            confidence: 0.80,
            model_agreement: 0.90,
            time_until_event_secs: 3600,
            liquidity: 50000.0,
            is_live: false,
            is_parlay: false,
        };

        let allocation = engine.kelly_allocation(&opportunity).unwrap();
        assert!(allocation > Decimal::ZERO);
        assert!(allocation < Decimal::from_str("10000.00").unwrap());
    }
}
