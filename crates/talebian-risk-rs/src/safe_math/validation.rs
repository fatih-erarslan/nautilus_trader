//! Input validation framework for financial data

use super::{SafeMathResult, is_valid_price, is_valid_volume, is_valid_ratio};
use crate::error::TalebianError;
use crate::types::MarketData;
use std::collections::HashMap;

/// Validation result with detailed error information
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }
    
    pub fn add_error(&mut self, error: String) {
        self.is_valid = false;
        self.errors.push(error);
    }
    
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    pub fn merge(&mut self, other: ValidationResult) {
        if !other.is_valid {
            self.is_valid = false;
        }
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// Comprehensive MarketData validation
pub fn validate_market_data(data: &MarketData) -> ValidationResult {
    let mut result = ValidationResult::new();
    
    // Validate price
    if !is_valid_price(data.price) {
        result.add_error(format!("Invalid price: {}", data.price));
    }
    
    // Validate volume
    if !is_valid_volume(data.volume) {
        result.add_error(format!("Invalid volume: {}", data.volume));
    }
    
    // Validate bid/ask
    if !is_valid_price(data.bid) {
        result.add_error(format!("Invalid bid price: {}", data.bid));
    }
    
    if !is_valid_price(data.ask) {
        result.add_error(format!("Invalid ask price: {}", data.ask));
    }
    
    // Validate bid <= ask
    if data.bid > data.ask {
        result.add_error(format!("Bid price ({}) > Ask price ({})", data.bid, data.ask));
    }
    
    // Validate spread isn't too wide
    let spread_ratio = (data.ask - data.bid) / data.price;
    if spread_ratio > 0.1 {
        result.add_warning(format!("Wide spread: {:.2}%", spread_ratio * 100.0));
    }
    
    // Validate bid/ask volumes
    if !is_valid_volume(data.bid_volume) {
        result.add_error(format!("Invalid bid volume: {}", data.bid_volume));
    }
    
    if !is_valid_volume(data.ask_volume) {
        result.add_error(format!("Invalid ask volume: {}", data.ask_volume));
    }
    
    // Validate volatility
    if !data.volatility.is_finite() || data.volatility < 0.0 {
        result.add_error(format!("Invalid volatility: {}", data.volatility));
    }
    
    if data.volatility > 5.0 {
        result.add_warning(format!("Very high volatility: {:.2}", data.volatility));
    }
    
    // Validate returns array
    let returns_validation = validate_returns_array(&data.returns);
    result.merge(returns_validation);
    
    // Validate volume history
    let volume_validation = validate_volume_array(&data.volume_history);
    result.merge(volume_validation);
    
    // Validate price consistency
    let mid_price = (data.bid + data.ask) / 2.0;
    let price_diff_ratio = (data.price - mid_price).abs() / data.price;
    if price_diff_ratio > 0.05 {
        result.add_warning(format!(
            "Price inconsistency: price={}, mid_price={}, diff={:.2}%",
            data.price, mid_price, price_diff_ratio * 100.0
        ));
    }
    
    result
}

/// Validate array of returns
pub fn validate_returns_array(returns: &[f64]) -> ValidationResult {
    let mut result = ValidationResult::new();
    
    if returns.is_empty() {
        result.add_warning("Empty returns array".to_string());
        return result;
    }
    
    for (i, &ret) in returns.iter().enumerate() {
        if !ret.is_finite() {
            result.add_error(format!("Invalid return at index {}: {}", i, ret));
        } else if ret < -1.0 {
            result.add_error(format!("Invalid return at index {} (< -100%): {}", i, ret));
        } else if ret > 10.0 {
            result.add_warning(format!("Extreme return at index {} (> 1000%): {}", i, ret));
        }
    }
    
    // Check for excessive outliers
    if returns.len() > 10 {
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = (returns.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64)
            .sqrt();
        
        let outlier_count = returns.iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * std_dev)
            .count();
        
        if outlier_count > returns.len() / 10 {
            result.add_warning(format!(
                "High number of outliers: {} out of {} returns",
                outlier_count, returns.len()
            ));
        }
    }
    
    result
}

/// Validate array of volumes
pub fn validate_volume_array(volumes: &[f64]) -> ValidationResult {
    let mut result = ValidationResult::new();
    
    if volumes.is_empty() {
        result.add_warning("Empty volume array".to_string());
        return result;
    }
    
    for (i, &vol) in volumes.iter().enumerate() {
        if !is_valid_volume(vol) {
            result.add_error(format!("Invalid volume at index {}: {}", i, vol));
        }
    }
    
    result
}

/// Validate allocation weights
pub fn validate_allocation_weights(allocation: &HashMap<String, f64>) -> SafeMathResult<()> {
    if allocation.is_empty() {
        return Err(TalebianError::data("Empty allocation"));
    }
    
    let mut total_weight = 0.0;
    
    for (asset, &weight) in allocation {
        if !weight.is_finite() {
            return Err(TalebianError::data(format!(
                "Invalid weight for {}: {}", asset, weight
            )));
        }
        
        if weight < 0.0 {
            return Err(TalebianError::data(format!(
                "Negative weight for {}: {}", asset, weight
            )));
        }
        
        if weight > 1.0 {
            return Err(TalebianError::data(format!(
                "Weight exceeds 100% for {}: {}", asset, weight
            )));
        }
        
        total_weight += weight;
    }
    
    if (total_weight - 1.0).abs() > 0.01 {
        return Err(TalebianError::data(format!(
            "Allocation weights sum to {:.4} instead of 1.0", total_weight
        )));
    }
    
    Ok(())
}

/// Validate price bounds for sanity
pub fn validate_price_bounds(price: f64, min_price: f64, max_price: f64) -> SafeMathResult<()> {
    if !is_valid_price(price) {
        return Err(TalebianError::data(format!("Invalid price: {}", price)));
    }
    
    if !is_valid_price(min_price) || !is_valid_price(max_price) {
        return Err(TalebianError::data("Invalid price bounds"));
    }
    
    if min_price >= max_price {
        return Err(TalebianError::data(
            "Minimum price must be less than maximum price"
        ));
    }
    
    if price < min_price {
        return Err(TalebianError::data(format!(
            "Price {} below minimum bound {}", price, min_price
        )));
    }
    
    if price > max_price {
        return Err(TalebianError::data(format!(
            "Price {} above maximum bound {}", price, max_price
        )));
    }
    
    Ok(())
}

/// Validate Kelly fraction parameters
pub fn validate_kelly_params(
    win_rate: f64,
    avg_win: f64,
    avg_loss: f64,
) -> SafeMathResult<()> {
    if !is_valid_ratio(win_rate) {
        return Err(TalebianError::data(format!(
            "Invalid win rate: {}", win_rate
        )));
    }
    
    if avg_win <= 0.0 || !avg_win.is_finite() {
        return Err(TalebianError::data(format!(
            "Invalid average win: {}", avg_win
        )));
    }
    
    if avg_loss <= 0.0 || !avg_loss.is_finite() {
        return Err(TalebianError::data(format!(
            "Invalid average loss: {}", avg_loss
        )));
    }
    
    Ok(())
}

/// Validate volatility parameters
pub fn validate_volatility_params(volatility: f64, lookback_period: usize) -> SafeMathResult<()> {
    if !volatility.is_finite() || volatility < 0.0 {
        return Err(TalebianError::data(format!(
            "Invalid volatility: {}", volatility
        )));
    }
    
    if volatility > 10.0 {
        return Err(TalebianError::data(format!(
            "Extremely high volatility: {}", volatility
        )));
    }
    
    if lookback_period == 0 {
        return Err(TalebianError::data("Lookback period cannot be zero"));
    }
    
    if lookback_period > 10000 {
        return Err(TalebianError::data(format!(
            "Lookback period too large: {}", lookback_period
        )));
    }
    
    Ok(())
}

/// Validate position size parameters
pub fn validate_position_size_params(
    capital: f64,
    risk_per_trade: f64,
    max_position_size: f64,
) -> SafeMathResult<()> {
    if !is_valid_price(capital) {
        return Err(TalebianError::data(format!(
            "Invalid capital: {}", capital
        )));
    }
    
    if !is_valid_ratio(risk_per_trade) {
        return Err(TalebianError::data(format!(
            "Invalid risk per trade: {}", risk_per_trade
        )));
    }
    
    if risk_per_trade > 0.1 {
        return Err(TalebianError::data(format!(
            "Risk per trade too high: {:.1}%", risk_per_trade * 100.0
        )));
    }
    
    if !is_valid_ratio(max_position_size) {
        return Err(TalebianError::data(format!(
            "Invalid max position size: {}", max_position_size
        )));
    }
    
    if max_position_size > 0.5 {
        return Err(TalebianError::data(format!(
            "Max position size too high: {:.1}%", max_position_size * 100.0
        )));
    }
    
    Ok(())
}

/// Comprehensive validation for trading configuration
pub fn validate_trading_config(
    kelly_fraction: f64,
    max_leverage: f64,
    stop_loss: f64,
    take_profit: f64,
) -> SafeMathResult<()> {
    // Validate Kelly fraction
    if !is_valid_ratio(kelly_fraction) {
        return Err(TalebianError::data(format!(
            "Invalid Kelly fraction: {}", kelly_fraction
        )));
    }
    
    if kelly_fraction > 0.25 {
        return Err(TalebianError::data(format!(
            "Kelly fraction too aggressive: {:.1}%", kelly_fraction * 100.0
        )));
    }
    
    // Validate leverage
    if !max_leverage.is_finite() || max_leverage <= 0.0 {
        return Err(TalebianError::data(format!(
            "Invalid max leverage: {}", max_leverage
        )));
    }
    
    if max_leverage > 10.0 {
        return Err(TalebianError::data(format!(
            "Max leverage too high: {}", max_leverage
        )));
    }
    
    // Validate stop loss
    if !is_valid_ratio(stop_loss) {
        return Err(TalebianError::data(format!(
            "Invalid stop loss: {}", stop_loss
        )));
    }
    
    if stop_loss > 0.2 {
        return Err(TalebianError::data(format!(
            "Stop loss too wide: {:.1}%", stop_loss * 100.0
        )));
    }
    
    // Validate take profit
    if !is_valid_ratio(take_profit) {
        return Err(TalebianError::data(format!(
            "Invalid take profit: {}", take_profit
        )));
    }
    
    // Risk-reward ratio check
    if take_profit < stop_loss {
        return Err(TalebianError::data(
            "Take profit should be greater than stop loss"
        ));
    }
    
    let risk_reward_ratio = take_profit / stop_loss;
    if risk_reward_ratio < 1.5 {
        return Err(TalebianError::data(format!(
            "Poor risk-reward ratio: {:.2}", risk_reward_ratio
        )));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_valid_market_data() -> MarketData {
        MarketData {
            timestamp: Utc::now(),
            timestamp_unix: 1000000000,
            price: 100.0,
            volume: 1000.0,
            bid: 99.5,
            ask: 100.5,
            bid_volume: 500.0,
            ask_volume: 500.0,
            volatility: 0.2,
            returns: vec![0.01, -0.02, 0.015, -0.01],
            volume_history: vec![900.0, 1100.0, 950.0, 1050.0],
        }
    }

    #[test]
    fn test_validate_market_data_valid() {
        let data = create_valid_market_data();
        let result = validate_market_data(&data);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validate_market_data_invalid_price() {
        let mut data = create_valid_market_data();
        data.price = -10.0;
        let result = validate_market_data(&data);
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_validate_market_data_bid_ask_inverted() {
        let mut data = create_valid_market_data();
        data.bid = 101.0;
        data.ask = 100.0;
        let result = validate_market_data(&data);
        assert!(!result.is_valid);
        assert!(result.errors.iter().any(|e| e.contains("Bid price")));
    }

    #[test]
    fn test_validate_allocation_weights() {
        let mut allocation = HashMap::new();
        allocation.insert("BTC".to_string(), 0.6);
        allocation.insert("ETH".to_string(), 0.4);
        
        assert!(validate_allocation_weights(&allocation).is_ok());
        
        allocation.insert("ADA".to_string(), 0.1); // Now sums to 1.1
        assert!(validate_allocation_weights(&allocation).is_err());
    }

    #[test]
    fn test_validate_kelly_params() {
        assert!(validate_kelly_params(0.6, 100.0, 50.0).is_ok());
        assert!(validate_kelly_params(1.5, 100.0, 50.0).is_err()); // Win rate > 1
        assert!(validate_kelly_params(0.6, -100.0, 50.0).is_err()); // Negative avg win
        assert!(validate_kelly_params(0.6, 100.0, -50.0).is_err()); // Negative avg loss
    }

    #[test]
    fn test_validate_trading_config() {
        assert!(validate_trading_config(0.2, 2.0, 0.02, 0.06).is_ok());
        assert!(validate_trading_config(0.5, 2.0, 0.02, 0.06).is_err()); // Kelly too high
        assert!(validate_trading_config(0.2, 20.0, 0.02, 0.06).is_err()); // Leverage too high
        assert!(validate_trading_config(0.2, 2.0, 0.06, 0.02).is_err()); // Bad risk-reward
    }
}