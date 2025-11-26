use anyhow::{anyhow, Result};

// Resource limit constants
/// Maximum size for JSON input (1MB)
pub const MAX_JSON_SIZE: usize = 1_000_000;

/// Maximum array length for input arrays
pub const MAX_ARRAY_LENGTH: usize = 10_000;

/// Maximum string length for input strings
pub const MAX_STRING_LENGTH: usize = 100_000;

/// Maximum number of agents in a swarm
pub const MAX_SWARM_AGENTS: u32 = 100;

/// Maximum concurrent requests
pub const MAX_CONCURRENT_REQUESTS: usize = 1000;

/// Maximum portfolio positions
pub const MAX_PORTFOLIO_POSITIONS: usize = 10_000;

/// Maximum backtest days (10 years)
pub const MAX_BACKTEST_DAYS: u32 = 3650;

/// Maximum neural network epochs
pub const MAX_NEURAL_EPOCHS: u32 = 10_000;

/// Maximum syndicate members
pub const MAX_SYNDICATE_MEMBERS: u32 = 1000;

/// Maximum batch size for operations
pub const MAX_BATCH_SIZE: usize = 1000;

/// Maximum number of symbols in a request
pub const MAX_SYMBOLS: usize = 100;

/// Maximum number of trades in a multi-asset trade
pub const MAX_TRADES_PER_REQUEST: usize = 50;

/// Maximum number of betting opportunities
pub const MAX_BETTING_OPPORTUNITIES: usize = 100;

/// Validate JSON string size
pub fn validate_json_size(json: &str, field: &str) -> Result<()> {
    if json.len() > MAX_JSON_SIZE {
        return Err(anyhow!(
            "{} exceeds maximum size of {} bytes (got {})",
            field, MAX_JSON_SIZE, json.len()
        ));
    }
    Ok(())
}

/// Validate array length
pub fn validate_array_length(len: usize, field: &str) -> Result<()> {
    if len > MAX_ARRAY_LENGTH {
        return Err(anyhow!(
            "{} exceeds maximum length of {} (got {})",
            field, MAX_ARRAY_LENGTH, len
        ));
    }
    Ok(())
}

/// Validate string length
pub fn validate_string_length(s: &str, field: &str) -> Result<()> {
    if s.len() > MAX_STRING_LENGTH {
        return Err(anyhow!(
            "{} exceeds maximum length of {} (got {})",
            field, MAX_STRING_LENGTH, s.len()
        ));
    }
    Ok(())
}

/// Validate number of swarm agents
pub fn validate_swarm_agents(count: u32, field: &str) -> Result<()> {
    if count > MAX_SWARM_AGENTS {
        return Err(anyhow!(
            "{} exceeds maximum of {} agents (got {})",
            field, MAX_SWARM_AGENTS, count
        ));
    }
    Ok(())
}

/// Validate portfolio positions count
pub fn validate_portfolio_positions(count: usize, field: &str) -> Result<()> {
    if count > MAX_PORTFOLIO_POSITIONS {
        return Err(anyhow!(
            "{} exceeds maximum of {} positions (got {})",
            field, MAX_PORTFOLIO_POSITIONS, count
        ));
    }
    Ok(())
}

/// Validate backtest days
pub fn validate_backtest_days(days: u32, field: &str) -> Result<()> {
    if days > MAX_BACKTEST_DAYS {
        return Err(anyhow!(
            "{} exceeds maximum of {} days (got {})",
            field, MAX_BACKTEST_DAYS, days
        ));
    }
    Ok(())
}

/// Validate neural epochs
pub fn validate_neural_epochs(epochs: u32, field: &str) -> Result<()> {
    if epochs > MAX_NEURAL_EPOCHS {
        return Err(anyhow!(
            "{} exceeds maximum of {} epochs (got {})",
            field, MAX_NEURAL_EPOCHS, epochs
        ));
    }
    Ok(())
}

/// Validate syndicate members
pub fn validate_syndicate_members(count: u32, field: &str) -> Result<()> {
    if count > MAX_SYNDICATE_MEMBERS {
        return Err(anyhow!(
            "{} exceeds maximum of {} members (got {})",
            field, MAX_SYNDICATE_MEMBERS, count
        ));
    }
    Ok(())
}

/// Validate batch size
pub fn validate_batch_size(size: usize, field: &str) -> Result<()> {
    if size > MAX_BATCH_SIZE {
        return Err(anyhow!(
            "{} exceeds maximum batch size of {} (got {})",
            field, MAX_BATCH_SIZE, size
        ));
    }
    Ok(())
}

/// Validate symbols count
pub fn validate_symbols_count(count: usize, field: &str) -> Result<()> {
    if count > MAX_SYMBOLS {
        return Err(anyhow!(
            "{} exceeds maximum of {} symbols (got {})",
            field, MAX_SYMBOLS, count
        ));
    }
    Ok(())
}

/// Validate trades count
pub fn validate_trades_count(count: usize, field: &str) -> Result<()> {
    if count > MAX_TRADES_PER_REQUEST {
        return Err(anyhow!(
            "{} exceeds maximum of {} trades per request (got {})",
            field, MAX_TRADES_PER_REQUEST, count
        ));
    }
    Ok(())
}

/// Validate betting opportunities count
pub fn validate_betting_opportunities(count: usize, field: &str) -> Result<()> {
    if count > MAX_BETTING_OPPORTUNITIES {
        return Err(anyhow!(
            "{} exceeds maximum of {} opportunities (got {})",
            field, MAX_BETTING_OPPORTUNITIES, count
        ));
    }
    Ok(())
}

/// Validate positive number
pub fn validate_positive(value: f64, field: &str) -> Result<()> {
    if value <= 0.0 {
        return Err(anyhow!("{} must be positive (got {})", field, value));
    }
    Ok(())
}

/// Validate non-negative number
pub fn validate_non_negative(value: f64, field: &str) -> Result<()> {
    if value < 0.0 {
        return Err(anyhow!("{} must be non-negative (got {})", field, value));
    }
    Ok(())
}

/// Validate percentage (0-1)
pub fn validate_percentage(value: f64, field: &str) -> Result<()> {
    if value < 0.0 || value > 1.0 {
        return Err(anyhow!("{} must be between 0 and 1 (got {})", field, value));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_json_size() {
        let small_json = "{}";
        assert!(validate_json_size(small_json, "test").is_ok());

        let large_json = "x".repeat(MAX_JSON_SIZE + 1);
        assert!(validate_json_size(&large_json, "test").is_err());
    }

    #[test]
    fn test_validate_array_length() {
        assert!(validate_array_length(100, "test").is_ok());
        assert!(validate_array_length(MAX_ARRAY_LENGTH + 1, "test").is_err());
    }

    #[test]
    fn test_validate_string_length() {
        let short_str = "hello";
        assert!(validate_string_length(short_str, "test").is_ok());

        let long_str = "x".repeat(MAX_STRING_LENGTH + 1);
        assert!(validate_string_length(&long_str, "test").is_err());
    }

    #[test]
    fn test_validate_swarm_agents() {
        assert!(validate_swarm_agents(10, "test").is_ok());
        assert!(validate_swarm_agents(MAX_SWARM_AGENTS + 1, "test").is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(0.0, "test").is_err());
        assert!(validate_positive(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_non_negative() {
        assert!(validate_non_negative(0.0, "test").is_ok());
        assert!(validate_non_negative(1.0, "test").is_ok());
        assert!(validate_non_negative(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_percentage() {
        assert!(validate_percentage(0.0, "test").is_ok());
        assert!(validate_percentage(0.5, "test").is_ok());
        assert!(validate_percentage(1.0, "test").is_ok());
        assert!(validate_percentage(-0.1, "test").is_err());
        assert!(validate_percentage(1.1, "test").is_err());
    }
}
