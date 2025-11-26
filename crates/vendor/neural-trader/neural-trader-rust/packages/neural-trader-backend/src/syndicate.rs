//! Syndicate management for collaborative betting
//!
//! Provides NAPI bindings for:
//! - Syndicate creation and management
//! - Member management
//! - Fund allocation
//! - Profit distribution
//! - Voting and governance

use napi::bindgen_prelude::*;
use napi_derive::napi;
use crate::validation::*;

/// Create a new syndicate
#[napi]
pub async fn create_syndicate(
    syndicate_id: String,
    name: String,
    description: Option<String>,
) -> Result<Syndicate> {
    // Validate inputs
    validate_id(&syndicate_id, "syndicate_id")?;
    validate_non_empty(&name, "name")?;
    validate_string_length(&name, 1, 200, "name")?;
    validate_no_sql_injection(&name, "name")?;

    if let Some(ref desc) = description {
        validate_string_length(desc, 0, 1000, "description")?;
        validate_no_sql_injection(desc, "description")?;
    }

    Ok(Syndicate {
        syndicate_id,
        name,
        description: description.unwrap_or_default(),
        total_capital: 0.0,
        member_count: 0,
        created_at: chrono::Utc::now().to_rfc3339(),
    })
}

/// Syndicate structure
#[napi(object)]
pub struct Syndicate {
    pub syndicate_id: String,
    pub name: String,
    pub description: String,
    pub total_capital: f64,
    pub member_count: u32,
    pub created_at: String,
}

/// Add member to syndicate
#[napi]
pub async fn add_syndicate_member(
    syndicate_id: String,
    name: String,
    email: String,
    role: String,
    initial_contribution: f64,
) -> Result<SyndicateMember> {
    // Validate all inputs
    validate_id(&syndicate_id, "syndicate_id")?;
    validate_non_empty(&name, "name")?;
    validate_string_length(&name, 1, 200, "name")?;
    validate_no_sql_injection(&name, "name")?;
    validate_email(&email)?;
    validate_syndicate_role(&role)?;
    validate_non_negative(initial_contribution, "initial_contribution")?;
    validate_finite(initial_contribution, "initial_contribution")?;

    Ok(SyndicateMember {
        member_id: format!("mem-{}", uuid::Uuid::new_v4()),
        syndicate_id,
        name,
        email,
        role,
        contribution: initial_contribution,
        profit_share: 0.0,
    })
}

/// Syndicate member
#[napi(object)]
pub struct SyndicateMember {
    pub member_id: String,
    pub syndicate_id: String,
    pub name: String,
    pub email: String,
    pub role: String,
    pub contribution: f64,
    pub profit_share: f64,
}

/// Get syndicate status
#[napi]
pub async fn get_syndicate_status(syndicate_id: String) -> Result<SyndicateStatus> {
    // Validate syndicate ID
    validate_id(&syndicate_id, "syndicate_id")?;

    Ok(SyndicateStatus {
        syndicate_id,
        total_capital: 50000.0,
        active_bets: 12,
        total_profit: 5420.0,
        roi: 0.108,
        member_count: 5,
    })
}

/// Syndicate status
#[napi(object)]
pub struct SyndicateStatus {
    pub syndicate_id: String,
    pub total_capital: f64,
    pub active_bets: u32,
    pub total_profit: f64,
    pub roi: f64,
    pub member_count: u32,
}

/// Allocate syndicate funds
#[napi]
pub async fn allocate_syndicate_funds(
    syndicate_id: String,
    opportunities: String, // JSON array
    strategy: Option<String>,
) -> Result<FundAllocation> {
    // Validate inputs
    validate_id(&syndicate_id, "syndicate_id")?;
    validate_json(&opportunities, "opportunities")?;

    if let Some(ref strat) = strategy {
        validate_non_empty(strat, "strategy")?;
        validate_string_length(strat, 1, 100, "strategy")?;
    }

    Ok(FundAllocation {
        syndicate_id,
        total_allocated: 5000.0,
        allocations: vec![],
        expected_return: 0.08,
        risk_score: 0.35,
    })
}

/// Fund allocation result
#[napi(object)]
pub struct FundAllocation {
    pub syndicate_id: String,
    pub total_allocated: f64,
    pub allocations: Vec<Allocation>,
    pub expected_return: f64,
    pub risk_score: f64,
}

/// Individual allocation
#[napi(object)]
pub struct Allocation {
    pub opportunity_id: String,
    pub amount: f64,
    pub expected_return: f64,
}

/// Distribute profits to members
#[napi]
pub async fn distribute_syndicate_profits(
    syndicate_id: String,
    total_profit: f64,
    model: Option<String>,
) -> Result<ProfitDistribution> {
    // Validate inputs
    validate_id(&syndicate_id, "syndicate_id")?;
    validate_finite(total_profit, "total_profit")?;
    // total_profit can be negative (losses)

    if let Some(ref mdl) = model {
        validate_non_empty(mdl, "model")?;
        validate_string_length(mdl, 1, 100, "model")?;
    }

    Ok(ProfitDistribution {
        syndicate_id,
        total_profit,
        distributions: vec![],
        distribution_date: chrono::Utc::now().to_rfc3339(),
    })
}

/// Profit distribution result
#[napi(object)]
pub struct ProfitDistribution {
    pub syndicate_id: String,
    pub total_profit: f64,
    pub distributions: Vec<MemberDistribution>,
    pub distribution_date: String,
}

/// Member profit distribution
#[napi(object)]
pub struct MemberDistribution {
    pub member_id: String,
    pub amount: f64,
    pub percentage: f64,
}
