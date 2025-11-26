//! Input validation utilities for NAPI bindings
//!
//! Provides comprehensive validation for all input parameters to ensure:
//! - Data integrity
//! - Security (prevent injection attacks)
//! - Business logic constraints
//! - Type safety beyond what Rust's type system provides

use crate::error::{validation_error, NeuralTraderError};
use chrono::Datelike;
use napi::bindgen_prelude::*;
use regex::Regex;
use std::sync::OnceLock;

// Compile regex patterns once
static SYMBOL_REGEX: OnceLock<Regex> = OnceLock::new();
static EMAIL_REGEX: OnceLock<Regex> = OnceLock::new();
static ISO8601_DATE_REGEX: OnceLock<Regex> = OnceLock::new();
static ALPHANUMERIC_REGEX: OnceLock<Regex> = OnceLock::new();

/// Initialize regex patterns (called on first use)
fn symbol_regex() -> &'static Regex {
    SYMBOL_REGEX.get_or_init(|| Regex::new(r"^[A-Z0-9]{1,10}$").unwrap())
}

fn email_regex() -> &'static Regex {
    EMAIL_REGEX.get_or_init(|| {
        Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap()
    })
}

fn iso8601_date_regex() -> &'static Regex {
    ISO8601_DATE_REGEX.get_or_init(|| {
        Regex::new(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?$").unwrap()
    })
}

fn alphanumeric_regex() -> &'static Regex {
    ALPHANUMERIC_REGEX.get_or_init(|| Regex::new(r"^[a-zA-Z0-9_-]+$").unwrap())
}

/// Valid trading actions
const VALID_ACTIONS: &[&str] = &["buy", "sell", "hold"];

/// Valid order types
const VALID_ORDER_TYPES: &[&str] = &["market", "limit", "stop", "stop-limit"];

/// Valid trading strategies
const VALID_STRATEGIES: &[&str] = &[
    "momentum",
    "mean_reversion",
    "pairs_trading",
    "market_making",
    "arbitrage",
    "trend_following",
    "neural_forecast",
];

/// Valid sports for betting
const VALID_SPORTS: &[&str] = &[
    "soccer",
    "basketball",
    "baseball",
    "football",
    "tennis",
    "hockey",
    "cricket",
    "rugby",
];

/// Valid syndicate roles
const VALID_SYNDICATE_ROLES: &[&str] = &[
    "owner",
    "admin",
    "member",
    "viewer",
];

/// Valid betting markets
const VALID_BETTING_MARKETS: &[&str] = &[
    "moneyline",
    "spread",
    "totals",
    "h2h",
];

/// Validate a trading symbol
///
/// Rules:
/// - Must be uppercase alphanumeric
/// - Length: 1-10 characters
/// - No special characters except numbers
pub fn validate_symbol(symbol: &str) -> Result<()> {
    if symbol.is_empty() {
        return Err(validation_error("Symbol cannot be empty"));
    }

    if symbol.len() > 10 {
        return Err(validation_error(format!(
            "Symbol too long: {} (max 10 characters)",
            symbol.len()
        )));
    }

    if !symbol_regex().is_match(symbol) {
        return Err(validation_error(format!(
            "Invalid symbol format: '{}'. Must be uppercase alphanumeric (1-10 chars)",
            symbol
        )));
    }

    Ok(())
}

/// Validate a date string (ISO 8601 format)
///
/// Accepts formats:
/// - YYYY-MM-DD
/// - YYYY-MM-DDTHH:MM:SS
/// - YYYY-MM-DDTHH:MM:SS.sssZ
/// - YYYY-MM-DDTHH:MM:SS+HH:MM
pub fn validate_date(date: &str, field_name: &str) -> Result<()> {
    if date.is_empty() {
        return Err(validation_error(format!("{} cannot be empty", field_name)));
    }

    if !iso8601_date_regex().is_match(date) {
        return Err(validation_error(format!(
            "Invalid {} date format: '{}'. Expected ISO 8601 (e.g., '2024-01-15' or '2024-01-15T10:30:00Z')",
            field_name, date
        )));
    }

    // Parse to verify it's a valid date
    match chrono::DateTime::parse_from_rfc3339(date) {
        Ok(parsed) => {
            // Check for reasonable date range (1970-2100)
            let year = parsed.date_naive().year();
            if year < 1970 || year > 2100 {
                return Err(validation_error(format!(
                    "Date year out of range: {} (must be 1970-2100)",
                    year
                )));
            }
            Ok(())
        }
        Err(_) => {
            // Try simple date format YYYY-MM-DD
            match chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d") {
                Ok(parsed) => {
                    let year = parsed.year();
                    if year < 1970 || year > 2100 {
                        return Err(validation_error(format!(
                            "Date year out of range: {} (must be 1970-2100)",
                            year
                        )));
                    }
                    Ok(())
                }
                Err(_) => Err(validation_error(format!(
                    "Could not parse {} date: '{}'",
                    field_name, date
                ))),
            }
        }
    }
}

/// Validate a positive number (> 0)
pub fn validate_positive<T: PartialOrd + std::fmt::Display>(
    value: T,
    field_name: &str,
) -> Result<()>
where
    T: From<u8>,
{
    if value <= T::from(0) {
        return Err(validation_error(format!(
            "{} must be greater than 0, got: {}",
            field_name, value
        )));
    }
    Ok(())
}

/// Validate a non-negative number (>= 0)
pub fn validate_non_negative<T: PartialOrd + std::fmt::Display>(
    value: T,
    field_name: &str,
) -> Result<()>
where
    T: From<u8>,
{
    if value < T::from(0) {
        return Err(validation_error(format!(
            "{} must be non-negative, got: {}",
            field_name, value
        )));
    }
    Ok(())
}

/// Validate a number is finite (not NaN or Infinity)
pub fn validate_finite(value: f64, field_name: &str) -> Result<()> {
    if !value.is_finite() {
        return Err(validation_error(format!(
            "{} must be a finite number (not NaN or Infinity)",
            field_name
        )));
    }
    Ok(())
}

/// Validate a probability (0.0 to 1.0 inclusive)
pub fn validate_probability(value: f64, field_name: &str) -> Result<()> {
    validate_finite(value, field_name)?;

    if value < 0.0 || value > 1.0 {
        return Err(validation_error(format!(
            "{} must be between 0.0 and 1.0, got: {}",
            field_name, value
        )));
    }
    Ok(())
}

/// Validate betting odds (must be > 1.0)
pub fn validate_odds(odds: f64, field_name: &str) -> Result<()> {
    validate_finite(odds, field_name)?;

    if odds <= 1.0 {
        return Err(validation_error(format!(
            "{} must be greater than 1.0 (decimal odds format), got: {}",
            field_name, odds
        )));
    }

    // Reasonable upper limit to prevent errors
    if odds > 1000.0 {
        return Err(validation_error(format!(
            "{} unreasonably high: {} (max 1000.0)",
            field_name, odds
        )));
    }

    Ok(())
}

/// Validate a number is within a range (inclusive)
pub fn validate_range<T: PartialOrd + std::fmt::Display>(
    value: T,
    min: T,
    max: T,
    field_name: &str,
) -> Result<()> {
    if value < min || value > max {
        return Err(validation_error(format!(
            "{} must be between {} and {}, got: {}",
            field_name, min, max, value
        )));
    }
    Ok(())
}

/// Validate a string length
pub fn validate_string_length(
    value: &str,
    min: usize,
    max: usize,
    field_name: &str,
) -> Result<()> {
    let len = value.len();
    if len < min || len > max {
        return Err(validation_error(format!(
            "{} length must be between {} and {} characters, got: {}",
            field_name, min, max, len
        )));
    }
    Ok(())
}

/// Validate a string is not empty
pub fn validate_non_empty(value: &str, field_name: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(validation_error(format!("{} cannot be empty", field_name)));
    }
    Ok(())
}

/// Validate a string contains no SQL injection patterns
pub fn validate_no_sql_injection(value: &str, field_name: &str) -> Result<()> {
    let lowercase = value.to_lowercase();
    let sql_keywords = ["select", "insert", "update", "delete", "drop", "union", "--", "/*", "*/"];

    for keyword in sql_keywords {
        if lowercase.contains(keyword) {
            return Err(validation_error(format!(
                "{} contains potentially dangerous SQL keyword: '{}'",
                field_name, keyword
            )));
        }
    }
    Ok(())
}

/// Validate a trading action
pub fn validate_action(action: &str) -> Result<()> {
    validate_non_empty(action, "action")?;

    let lowercase_action = action.to_lowercase();
    if !VALID_ACTIONS.contains(&lowercase_action.as_str()) {
        return Err(validation_error(format!(
            "Invalid action: '{}'. Must be one of: {}",
            action,
            VALID_ACTIONS.join(", ")
        )));
    }
    Ok(())
}

/// Validate an order type
pub fn validate_order_type(order_type: &str) -> Result<()> {
    validate_non_empty(order_type, "order_type")?;

    let lowercase = order_type.to_lowercase();
    if !VALID_ORDER_TYPES.contains(&lowercase.as_str()) {
        return Err(validation_error(format!(
            "Invalid order type: '{}'. Must be one of: {}",
            order_type,
            VALID_ORDER_TYPES.join(", ")
        )));
    }
    Ok(())
}

/// Validate a trading strategy name
pub fn validate_strategy(strategy: &str) -> Result<()> {
    validate_non_empty(strategy, "strategy")?;

    let lowercase = strategy.to_lowercase();
    if !VALID_STRATEGIES.contains(&lowercase.as_str()) {
        return Err(validation_error(format!(
            "Invalid strategy: '{}'. Must be one of: {}",
            strategy,
            VALID_STRATEGIES.join(", ")
        )));
    }
    Ok(())
}

/// Validate a sport name
pub fn validate_sport(sport: &str) -> Result<()> {
    validate_non_empty(sport, "sport")?;

    let lowercase = sport.to_lowercase();
    if !VALID_SPORTS.contains(&lowercase.as_str()) {
        return Err(validation_error(format!(
            "Invalid sport: '{}'. Must be one of: {}",
            sport,
            VALID_SPORTS.join(", ")
        )));
    }
    Ok(())
}

/// Validate an email address
pub fn validate_email(email: &str) -> Result<()> {
    validate_non_empty(email, "email")?;

    if !email_regex().is_match(email) {
        return Err(validation_error(format!(
            "Invalid email format: '{}'",
            email
        )));
    }

    // Additional checks
    if email.len() > 255 {
        return Err(validation_error("Email too long (max 255 characters)"));
    }

    Ok(())
}

/// Validate a syndicate role
pub fn validate_syndicate_role(role: &str) -> Result<()> {
    validate_non_empty(role, "role")?;

    let lowercase = role.to_lowercase();
    if !VALID_SYNDICATE_ROLES.contains(&lowercase.as_str()) {
        return Err(validation_error(format!(
            "Invalid syndicate role: '{}'. Must be one of: {}",
            role,
            VALID_SYNDICATE_ROLES.join(", ")
        )));
    }
    Ok(())
}

/// Validate an ID string (alphanumeric, underscores, hyphens only)
pub fn validate_id(id: &str, field_name: &str) -> Result<()> {
    validate_non_empty(id, field_name)?;
    validate_string_length(id, 1, 100, field_name)?;

    if !alphanumeric_regex().is_match(id) {
        return Err(validation_error(format!(
            "{} must contain only alphanumeric characters, underscores, and hyphens",
            field_name
        )));
    }
    Ok(())
}

/// Validate a betting market type
pub fn validate_betting_market(market: &str) -> Result<()> {
    validate_non_empty(market, "market")?;

    let lowercase = market.to_lowercase();
    if !VALID_BETTING_MARKETS.contains(&lowercase.as_str()) {
        return Err(validation_error(format!(
            "Invalid betting market: '{}'. Must be one of: {}",
            market,
            VALID_BETTING_MARKETS.join(", ")
        )));
    }
    Ok(())
}

/// Validate a JSON string can be parsed
pub fn validate_json(json: &str, field_name: &str) -> Result<()> {
    validate_non_empty(json, field_name)?;

    serde_json::from_str::<serde_json::Value>(json).map_err(|e| {
        validation_error(format!("Invalid JSON in {}: {}", field_name, e))
    })?;

    Ok(())
}

/// Validate limit price for limit orders
pub fn validate_limit_price(limit_price: Option<f64>, order_type: &str) -> Result<()> {
    if order_type == "limit" || order_type == "stop-limit" {
        match limit_price {
            Some(price) => {
                validate_positive(price, "limit_price")?;
                validate_finite(price, "limit_price")?;
            }
            None => {
                return Err(validation_error(format!(
                    "limit_price is required for order type '{}'",
                    order_type
                )));
            }
        }
    }
    Ok(())
}

/// Validate date range (start_date must be before end_date)
pub fn validate_date_range(start_date: &str, end_date: &str) -> Result<()> {
    validate_date(start_date, "start_date")?;
    validate_date(end_date, "end_date")?;

    // Parse and compare dates
    let start = chrono::DateTime::parse_from_rfc3339(start_date)
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(start_date, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc().fixed_offset())
        })
        .map_err(|_| validation_error("Could not parse start_date"))?;

    let end = chrono::DateTime::parse_from_rfc3339(end_date)
        .or_else(|_| {
            chrono::NaiveDate::parse_from_str(end_date, "%Y-%m-%d")
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc().fixed_offset())
        })
        .map_err(|_| validation_error("Could not parse end_date"))?;

    if start >= end {
        return Err(validation_error(format!(
            "start_date ({}) must be before end_date ({})",
            start_date, end_date
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_symbol() {
        assert!(validate_symbol("AAPL").is_ok());
        assert!(validate_symbol("BTC").is_ok());
        assert!(validate_symbol("SPY500").is_ok());

        assert!(validate_symbol("").is_err());
        assert!(validate_symbol("aapl").is_err()); // lowercase
        assert!(validate_symbol("AAPL-USD").is_err()); // special char
        assert!(validate_symbol("VERYLONGSYMBOL").is_err()); // too long
    }

    #[test]
    fn test_validate_email() {
        assert!(validate_email("user@example.com").is_ok());
        assert!(validate_email("test.user+tag@domain.co.uk").is_ok());

        assert!(validate_email("").is_err());
        assert!(validate_email("not-an-email").is_err());
        assert!(validate_email("@example.com").is_err());
        assert!(validate_email("user@").is_err());
    }

    #[test]
    fn test_validate_probability() {
        assert!(validate_probability(0.0, "prob").is_ok());
        assert!(validate_probability(0.5, "prob").is_ok());
        assert!(validate_probability(1.0, "prob").is_ok());

        assert!(validate_probability(-0.1, "prob").is_err());
        assert!(validate_probability(1.1, "prob").is_err());
        assert!(validate_probability(f64::NAN, "prob").is_err());
        assert!(validate_probability(f64::INFINITY, "prob").is_err());
    }

    #[test]
    fn test_validate_odds() {
        assert!(validate_odds(1.5, "odds").is_ok());
        assert!(validate_odds(2.0, "odds").is_ok());
        assert!(validate_odds(10.0, "odds").is_ok());

        assert!(validate_odds(1.0, "odds").is_err()); // must be > 1.0
        assert!(validate_odds(0.5, "odds").is_err());
        assert!(validate_odds(1001.0, "odds").is_err()); // too high
        assert!(validate_odds(f64::NAN, "odds").is_err());
    }

    #[test]
    fn test_validate_action() {
        assert!(validate_action("buy").is_ok());
        assert!(validate_action("BUY").is_ok()); // case insensitive
        assert!(validate_action("sell").is_ok());

        assert!(validate_action("invalid").is_err());
        assert!(validate_action("").is_err());
    }

    #[test]
    fn test_validate_strategy() {
        assert!(validate_strategy("momentum").is_ok());
        assert!(validate_strategy("MOMENTUM").is_ok()); // case insensitive
        assert!(validate_strategy("mean_reversion").is_ok());

        assert!(validate_strategy("invalid_strategy").is_err());
        assert!(validate_strategy("").is_err());
    }

    #[test]
    fn test_validate_date() {
        assert!(validate_date("2024-01-15", "test").is_ok());
        assert!(validate_date("2024-01-15T10:30:00Z", "test").is_ok());
        assert!(validate_date("2024-01-15T10:30:00+05:30", "test").is_ok());

        assert!(validate_date("2024-13-01", "test").is_err()); // invalid month
        assert!(validate_date("not-a-date", "test").is_err());
        assert!(validate_date("", "test").is_err());
    }

    #[test]
    fn test_validate_date_range() {
        assert!(validate_date_range("2024-01-01", "2024-12-31").is_ok());

        assert!(validate_date_range("2024-12-31", "2024-01-01").is_err()); // reversed
        assert!(validate_date_range("2024-01-01", "2024-01-01").is_err()); // same date
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1, "test").is_ok());
        assert!(validate_positive(100, "test").is_ok());
        assert!(validate_positive(0.1, "test").is_ok());

        assert!(validate_positive(0, "test").is_err());
        assert!(validate_positive(-1, "test").is_err());
    }

    #[test]
    fn test_validate_sql_injection() {
        assert!(validate_no_sql_injection("normal text", "test").is_ok());
        assert!(validate_no_sql_injection("Company Name Inc.", "test").is_ok());

        assert!(validate_no_sql_injection("'; DROP TABLE users--", "test").is_err());
        assert!(validate_no_sql_injection("SELECT * FROM", "test").is_err());
        assert!(validate_no_sql_injection("UNION SELECT password", "test").is_err());
    }
}
