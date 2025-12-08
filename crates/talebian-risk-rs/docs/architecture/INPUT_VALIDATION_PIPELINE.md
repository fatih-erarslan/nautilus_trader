# Input Validation Pipeline Architecture
## Multi-Layer Security Validation for Financial Trading Systems

**Document Version**: 1.0  
**Date**: August 16, 2025  
**Classification**: TECHNICAL SPECIFICATION  

---

## OVERVIEW

This specification defines a comprehensive, multi-layer input validation pipeline that protects the financial trading system from malicious inputs, data corruption, and calculation errors through progressive validation stages.

## VALIDATION ARCHITECTURE

### Pipeline Overview

```
Input Data
    ↓
┌─────────────────────┐
│   LAYER 1: SYNTAX   │ ← Basic structure, types, nulls
│    VALIDATION       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  LAYER 2: SEMANTIC  │ ← Relationships, consistency
│    VALIDATION       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ LAYER 3: BUSINESS   │ ← Domain rules, limits
│     LOGIC           │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ LAYER 4: SECURITY   │ ← Injection, tampering
│    VALIDATION       │
└─────────────────────┘
    ↓
┌─────────────────────┐
│ LAYER 5: COMPLIANCE │ ← Regulatory requirements
│    VALIDATION       │
└─────────────────────┘
    ↓
Validated Data
```

## LAYER 1: SYNTAX VALIDATION

### Core Syntax Validator

```rust
/// Layer 1: Syntax validation for basic data integrity
pub struct SyntaxValidator {
    config: SyntaxValidationConfig,
    field_validators: HashMap<String, Box<dyn FieldValidator>>,
    type_checkers: HashMap<String, Box<dyn TypeChecker>>,
}

impl SyntaxValidator {
    /// Validate market data structure and basic types
    pub fn validate_market_data(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check required fields presence
        self.validate_required_fields(data)?;
        
        // Validate numeric field integrity
        self.validate_numeric_fields(data)?;
        
        // Validate data structure consistency
        self.validate_data_structure(data)?;
        
        // Check array/vector sizes
        self.validate_collection_sizes(data)?;
        
        // Validate timestamps
        self.validate_timestamp_fields(data)?;
        
        Ok(())
    }
    
    /// Check all required fields are present and not null
    fn validate_required_fields(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        let required_fields = [
            ("price", data.price),
            ("volume", data.volume),
            ("bid", data.bid),
            ("ask", data.ask),
            ("timestamp", data.timestamp as f64),
        ];
        
        for (field_name, value) in required_fields {
            if value.is_nan() {
                return Err(TalebianSecureError::InputValidation {
                    field: field_name.to_string(),
                    reason: "Required field is NaN".to_string(),
                    source: None,
                });
            }
        }
        
        // Check collections are not empty when required
        if data.returns.is_empty() && self.config.require_returns {
            return Err(TalebianSecureError::InputValidation {
                field: "returns".to_string(),
                reason: "Returns array cannot be empty".to_string(),
                source: None,
            });
        }
        
        if data.volume_history.is_empty() && self.config.require_volume_history {
            return Err(TalebianSecureError::InputValidation {
                field: "volume_history".to_string(),
                reason: "Volume history cannot be empty".to_string(),
                source: None,
            });
        }
        
        Ok(())
    }
    
    /// Validate numeric fields for NaN, Infinity, and basic sanity
    fn validate_numeric_fields(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Primary price fields
        self.validate_floating_point("price", data.price)?;
        self.validate_floating_point("volume", data.volume)?;
        self.validate_floating_point("bid", data.bid)?;
        self.validate_floating_point("ask", data.ask)?;
        self.validate_floating_point("volatility", data.volatility)?;
        
        // Volume fields
        if let Some(bid_vol) = data.bid_volume {
            self.validate_floating_point("bid_volume", bid_vol)?;
        }
        if let Some(ask_vol) = data.ask_volume {
            self.validate_floating_point("ask_volume", ask_vol)?;
        }
        
        // Array fields
        for (i, &return_val) in data.returns.iter().enumerate() {
            self.validate_floating_point(&format!("returns[{}]", i), return_val)?;
        }
        
        for (i, &volume_val) in data.volume_history.iter().enumerate() {
            self.validate_floating_point(&format!("volume_history[{}]", i), volume_val)?;
        }
        
        Ok(())
    }
    
    /// Validate individual floating point value
    fn validate_floating_point(&self, field_name: &str, value: f64) -> Result<(), TalebianSecureError> {
        if value.is_nan() {
            return Err(TalebianSecureError::InvalidFloatingPoint {
                value: "NaN".to_string(),
                context: field_name.to_string(),
                operation_type: "input_validation".to_string(),
                input_summary: format!("Field: {}", field_name),
            });
        }
        
        if value.is_infinite() {
            return Err(TalebianSecureError::InvalidFloatingPoint {
                value: if value.is_sign_positive() { "Infinity" } else { "-Infinity" }.to_string(),
                context: field_name.to_string(),
                operation_type: "input_validation".to_string(),
                input_summary: format!("Field: {}", field_name),
            });
        }
        
        // Check for subnormal numbers that might cause precision issues
        if value != 0.0 && value.abs() < f64::MIN_POSITIVE {
            return Err(TalebianSecureError::PrecisionLoss {
                operation: format!("input_validation_{}", field_name),
                loss_percentage: 100.0,
                original_precision: 64,
                resulting_precision: 0,
            });
        }
        
        Ok(())
    }
    
    /// Validate data structure consistency
    fn validate_data_structure(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check array lengths are reasonable
        if data.returns.len() > self.config.max_returns_length {
            return Err(TalebianSecureError::InputValidation {
                field: "returns".to_string(),
                reason: format!("Returns array too long: {} > {}", 
                              data.returns.len(), 
                              self.config.max_returns_length),
                source: None,
            });
        }
        
        if data.volume_history.len() > self.config.max_volume_history_length {
            return Err(TalebianSecureError::InputValidation {
                field: "volume_history".to_string(),
                reason: format!("Volume history too long: {} > {}", 
                              data.volume_history.len(), 
                              self.config.max_volume_history_length),
                source: None,
            });
        }
        
        // Check for consistent array lengths if required
        if self.config.require_consistent_array_lengths {
            if !data.returns.is_empty() && !data.volume_history.is_empty() {
                if data.returns.len() != data.volume_history.len() {
                    return Err(TalebianSecureError::InputValidation {
                        field: "array_consistency".to_string(),
                        reason: format!("Inconsistent array lengths: returns={}, volume_history={}", 
                                      data.returns.len(), 
                                      data.volume_history.len()),
                        source: None,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate timestamp fields
    fn validate_timestamp_fields(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check timestamp is reasonable (not too far in past or future)
        let current_time = chrono::Utc::now().timestamp();
        let data_time = data.timestamp;
        
        // Check if timestamp is too old (more than 1 year)
        if data_time < current_time - 365 * 24 * 3600 {
            return Err(TalebianSecureError::TimestampInvalid {
                timestamp: data_time,
                reason: "Timestamp too far in past".to_string(),
                expected_range: Some((current_time - 365 * 24 * 3600, current_time + 3600)),
            });
        }
        
        // Check if timestamp is too far in future (more than 1 hour)
        if data_time > current_time + 3600 {
            return Err(TalebianSecureError::TimestampInvalid {
                timestamp: data_time,
                reason: "Timestamp too far in future".to_string(),
                expected_range: Some((current_time - 365 * 24 * 3600, current_time + 3600)),
            });
        }
        
        // Additional timestamp_unix field validation if present
        if let Some(timestamp_unix) = data.timestamp_unix {
            let diff = (timestamp_unix - data_time).abs();
            if diff > 60 { // Allow 1 minute difference
                return Err(TalebianSecureError::TimestampInvalid {
                    timestamp: timestamp_unix,
                    reason: format!("Timestamp fields inconsistent: diff={} seconds", diff),
                    expected_range: Some((data_time - 60, data_time + 60)),
                });
            }
        }
        
        Ok(())
    }
}

/// Configuration for syntax validation
#[derive(Debug, Clone)]
pub struct SyntaxValidationConfig {
    pub require_returns: bool,
    pub require_volume_history: bool,
    pub max_returns_length: usize,
    pub max_volume_history_length: usize,
    pub require_consistent_array_lengths: bool,
    pub allow_zero_values: bool,
    pub allow_negative_prices: bool,
    pub max_decimal_places: u8,
}

impl Default for SyntaxValidationConfig {
    fn default() -> Self {
        Self {
            require_returns: true,
            require_volume_history: true,
            max_returns_length: 10000,
            max_volume_history_length: 10000,
            require_consistent_array_lengths: false,
            allow_zero_values: false,
            allow_negative_prices: false,
            max_decimal_places: 8,
        }
    }
}
```

## LAYER 2: SEMANTIC VALIDATION

### Semantic Relationship Validator

```rust
/// Layer 2: Semantic validation for data relationships
pub struct SemanticValidator {
    config: SemanticValidationConfig,
    historical_validator: HistoricalDataValidator,
    relationship_checker: RelationshipChecker,
}

impl SemanticValidator {
    /// Validate semantic relationships in market data
    pub fn validate_market_data(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Validate price relationships
        self.validate_price_relationships(data)?;
        
        // Validate volume consistency
        self.validate_volume_consistency(data)?;
        
        // Validate temporal consistency
        self.validate_temporal_consistency(data)?;
        
        // Validate statistical relationships
        self.validate_statistical_relationships(data)?;
        
        // Cross-reference with historical patterns
        self.validate_historical_consistency(data)?;
        
        Ok(())
    }
    
    /// Validate price relationships (bid <= price <= ask, spreads, etc.)
    fn validate_price_relationships(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Basic bid-ask relationship
        if data.bid > data.ask {
            return Err(TalebianSecureError::InputValidation {
                field: "bid_ask_relationship".to_string(),
                reason: format!("Bid {} cannot be greater than ask {}", data.bid, data.ask),
                source: None,
            });
        }
        
        // Price within bid-ask spread
        if data.price < data.bid || data.price > data.ask {
            return Err(TalebianSecureError::InputValidation {
                field: "price_spread_relationship".to_string(),
                reason: format!("Price {} not within bid-ask spread [{}, {}]", 
                              data.price, data.bid, data.ask),
                source: None,
            });
        }
        
        // Check for reasonable spread
        let spread = data.ask - data.bid;
        let spread_pct = spread / data.price;
        
        if spread_pct > self.config.max_spread_percentage {
            return Err(TalebianSecureError::DataIntegrity {
                details: format!("Unusually wide spread: {:.4}% > {:.4}%", 
                               spread_pct * 100.0, 
                               self.config.max_spread_percentage * 100.0),
                data_hash: None,
                expected_hash: None,
            });
        }
        
        // Check for zero spread (might indicate stale data)
        if spread < f64::EPSILON && !self.config.allow_zero_spread {
            return Err(TalebianSecureError::DataIntegrity {
                details: "Zero bid-ask spread detected - possible stale data".to_string(),
                data_hash: None,
                expected_hash: None,
            });
        }
        
        // Validate price precision consistency
        let price_decimals = count_decimal_places(data.price);
        let bid_decimals = count_decimal_places(data.bid);
        let ask_decimals = count_decimal_places(data.ask);
        
        let max_decimals = price_decimals.max(bid_decimals).max(ask_decimals);
        if max_decimals > self.config.max_price_decimal_places {
            return Err(TalebianSecureError::InputValidation {
                field: "price_precision".to_string(),
                reason: format!("Price precision too high: {} decimal places", max_decimals),
                source: None,
            });
        }
        
        Ok(())
    }
    
    /// Validate volume consistency and relationships
    fn validate_volume_consistency(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check volume is positive
        if data.volume <= 0.0 && !self.config.allow_zero_volume {
            return Err(TalebianSecureError::VolumeInvalid {
                volume: data.volume,
                reason: "Volume must be positive".to_string(),
                market_symbol: data.symbol.clone().unwrap_or_default(),
                historical_avg: None,
            });
        }
        
        // Validate bid/ask volume relationships if present
        if let (Some(bid_vol), Some(ask_vol)) = (data.bid_volume, data.ask_volume) {
            let total_book_volume = bid_vol + ask_vol;
            
            // Check if current volume is reasonable compared to book
            if data.volume > total_book_volume * self.config.max_volume_to_book_ratio {
                return Err(TalebianSecureError::VolumeInvalid {
                    volume: data.volume,
                    reason: format!("Trade volume {} exceeds reasonable ratio of book volume {}", 
                                  data.volume, total_book_volume),
                    market_symbol: data.symbol.clone().unwrap_or_default(),
                    historical_avg: Some(total_book_volume),
                });
            }
            
            // Check for extreme volume imbalances
            let volume_imbalance = (bid_vol - ask_vol).abs() / (bid_vol + ask_vol);
            if volume_imbalance > self.config.max_volume_imbalance_ratio {
                return Err(TalebianSecureError::DataIntegrity {
                    details: format!("Extreme volume imbalance: {:.2}%", volume_imbalance * 100.0),
                    data_hash: None,
                    expected_hash: None,
                });
            }
        }
        
        // Validate volume history consistency
        if !data.volume_history.is_empty() {
            let recent_avg = data.volume_history.iter().sum::<f64>() / data.volume_history.len() as f64;
            
            // Check if current volume is extremely different from recent history
            let volume_deviation = (data.volume - recent_avg).abs() / recent_avg;
            if volume_deviation > self.config.max_volume_deviation_ratio {
                return Err(TalebianSecureError::VolumeInvalid {
                    volume: data.volume,
                    reason: format!("Volume deviation too high: {:.2}% from recent average", 
                                  volume_deviation * 100.0),
                    market_symbol: data.symbol.clone().unwrap_or_default(),
                    historical_avg: Some(recent_avg),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate temporal consistency and ordering
    fn validate_temporal_consistency(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check that arrays are in temporal order if timestamps are available
        if let Some(timestamps) = &data.timestamp_history {
            for i in 1..timestamps.len() {
                if timestamps[i] <= timestamps[i-1] {
                    return Err(TalebianSecureError::TimestampInvalid {
                        timestamp: timestamps[i],
                        reason: format!("Timestamps not in ascending order at index {}", i),
                        expected_range: Some((timestamps[i-1] + 1, i64::MAX)),
                    });
                }
            }
        }
        
        // Validate that current timestamp is consistent with any historical data
        if !data.volume_history.is_empty() && data.volume_history.len() > 1 {
            // Estimate time intervals and check for reasonableness
            let estimated_interval = self.estimate_data_interval(data)?;
            
            if estimated_interval < self.config.min_data_interval_seconds {
                return Err(TalebianSecureError::TimestampInvalid {
                    timestamp: data.timestamp,
                    reason: format!("Data interval too short: {} seconds", estimated_interval),
                    expected_range: Some((self.config.min_data_interval_seconds, i64::MAX)),
                });
            }
            
            if estimated_interval > self.config.max_data_interval_seconds {
                return Err(TalebianSecureError::TimestampInvalid {
                    timestamp: data.timestamp,
                    reason: format!("Data interval too long: {} seconds", estimated_interval),
                    expected_range: Some((0, self.config.max_data_interval_seconds)),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate statistical relationships (volatility, returns, etc.)
    fn validate_statistical_relationships(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Validate volatility is reasonable
        if data.volatility < 0.0 {
            return Err(TalebianSecureError::InputValidation {
                field: "volatility".to_string(),
                reason: "Volatility cannot be negative".to_string(),
                source: None,
            });
        }
        
        if data.volatility > self.config.max_volatility {
            return Err(TalebianSecureError::InputValidation {
                field: "volatility".to_string(),
                reason: format!("Volatility {} exceeds maximum {}", data.volatility, self.config.max_volatility),
                source: None,
            });
        }
        
        // Validate returns are reasonable if present
        if !data.returns.is_empty() {
            for (i, &return_val) in data.returns.iter().enumerate() {
                if return_val.abs() > self.config.max_single_period_return {
                    return Err(TalebianSecureError::InputValidation {
                        field: format!("returns[{}]", i),
                        reason: format!("Return {} exceeds maximum single period return {}", 
                                      return_val, self.config.max_single_period_return),
                        source: None,
                    });
                }
            }
            
            // Check if volatility is consistent with returns
            let calculated_volatility = calculate_volatility(&data.returns);
            let volatility_difference = (data.volatility - calculated_volatility).abs() / data.volatility;
            
            if volatility_difference > self.config.max_volatility_inconsistency {
                return Err(TalebianSecureError::DataIntegrity {
                    details: format!("Volatility inconsistent with returns: provided={:.4}, calculated={:.4}", 
                                   data.volatility, calculated_volatility),
                    data_hash: None,
                    expected_hash: None,
                });
            }
        }
        
        Ok(())
    }
    
    /// Estimate data interval from historical patterns
    fn estimate_data_interval(&self, data: &MarketData) -> Result<i64, TalebianSecureError> {
        // Implementation would analyze patterns in historical data
        // For now, return a reasonable default
        Ok(60) // 1 minute default
    }
}

/// Helper function to count decimal places
fn count_decimal_places(value: f64) -> u8 {
    let s = format!("{}", value);
    if let Some(pos) = s.find('.') {
        (s.len() - pos - 1) as u8
    } else {
        0
    }
}

/// Helper function to calculate volatility from returns
fn calculate_volatility(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|&r| (r - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    
    variance.sqrt()
}

/// Configuration for semantic validation
#[derive(Debug, Clone)]
pub struct SemanticValidationConfig {
    pub max_spread_percentage: f64,
    pub allow_zero_spread: bool,
    pub max_price_decimal_places: u8,
    pub allow_zero_volume: bool,
    pub max_volume_to_book_ratio: f64,
    pub max_volume_imbalance_ratio: f64,
    pub max_volume_deviation_ratio: f64,
    pub min_data_interval_seconds: i64,
    pub max_data_interval_seconds: i64,
    pub max_volatility: f64,
    pub max_single_period_return: f64,
    pub max_volatility_inconsistency: f64,
}

impl Default for SemanticValidationConfig {
    fn default() -> Self {
        Self {
            max_spread_percentage: 0.05,  // 5% max spread
            allow_zero_spread: false,
            max_price_decimal_places: 8,
            allow_zero_volume: false,
            max_volume_to_book_ratio: 10.0,
            max_volume_imbalance_ratio: 0.9,  // 90% imbalance max
            max_volume_deviation_ratio: 5.0,  // 500% deviation max
            min_data_interval_seconds: 1,
            max_data_interval_seconds: 86400,  // 1 day max
            max_volatility: 5.0,  // 500% volatility max
            max_single_period_return: 0.5,  // 50% single period return max
            max_volatility_inconsistency: 0.2,  // 20% inconsistency max
        }
    }
}
```

## LAYER 3: BUSINESS LOGIC VALIDATION

### Business Rules Validator

```rust
/// Layer 3: Business logic validation for domain-specific rules
pub struct BusinessLogicValidator {
    config: BusinessValidationConfig,
    market_hours_checker: MarketHoursChecker,
    trading_rules_engine: TradingRulesEngine,
    risk_limits_validator: RiskLimitsValidator,
}

impl BusinessLogicValidator {
    /// Validate business logic rules
    pub fn validate_market_data(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check trading hours
        self.validate_trading_hours(data)?;
        
        // Validate market conditions
        self.validate_market_conditions(data)?;
        
        // Check volatility bounds
        self.validate_volatility_bounds(data)?;
        
        // Validate liquidity requirements
        self.validate_liquidity_requirements(data)?;
        
        // Check risk parameters
        self.validate_risk_parameters(data)?;
        
        Ok(())
    }
    
    /// Validate trading hours and market sessions
    fn validate_trading_hours(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        let market_time = chrono::DateTime::from_timestamp(data.timestamp, 0)
            .ok_or_else(|| TalebianSecureError::TimestampInvalid {
                timestamp: data.timestamp,
                reason: "Invalid timestamp for market hours check".to_string(),
                expected_range: None,
            })?;
        
        // Get market symbol for specific market hours
        let symbol = data.symbol.as_ref().unwrap_or(&"DEFAULT".to_string());
        
        if !self.market_hours_checker.is_market_open(symbol, &market_time)? {
            // Check if this is allowed for after-hours data
            if !self.config.allow_after_hours_data {
                return Err(TalebianSecureError::InputValidation {
                    field: "market_hours".to_string(),
                    reason: format!("Market {} is closed at {}", symbol, market_time),
                    source: None,
                });
            }
            
            // Apply after-hours validation rules
            self.validate_after_hours_data(data, &market_time)?;
        }
        
        Ok(())
    }
    
    /// Validate market conditions for reasonableness
    fn validate_market_conditions(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check for market crash scenarios
        if !data.returns.is_empty() {
            let max_negative_return = data.returns.iter()
                .filter(|&&r| r < 0.0)
                .map(|&r| r.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            
            if max_negative_return > self.config.market_crash_threshold {
                return Err(TalebianSecureError::MarketRegimeChange {
                    details: format!("Potential market crash detected: {:.2}% negative return", 
                                   max_negative_return * 100.0),
                    previous_regime: "normal".to_string(),
                    current_regime: "crash".to_string(),
                    confidence: 0.8,
                });
            }
        }
        
        // Check for unusual price movements
        if let Some(symbol) = &data.symbol {
            if let Some(previous_price) = self.get_previous_price(symbol) {
                let price_change = (data.price - previous_price) / previous_price;
                
                if price_change.abs() > self.config.max_price_movement_threshold {
                    return Err(TalebianSecureError::DataIntegrity {
                        details: format!("Unusual price movement: {:.2}% for {}", 
                                       price_change * 100.0, symbol),
                        data_hash: None,
                        expected_hash: None,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate volatility bounds for business rules
    fn validate_volatility_bounds(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check minimum volatility (market might be stale)
        if data.volatility < self.config.min_volatility_threshold {
            return Err(TalebianSecureError::DataIntegrity {
                details: format!("Volatility too low: {:.6} < {:.6} - possible stale market", 
                               data.volatility, self.config.min_volatility_threshold),
                data_hash: None,
                expected_hash: None,
            });
        }
        
        // Check maximum volatility (extreme conditions)
        if data.volatility > self.config.max_volatility_threshold {
            return Err(TalebianSecureError::RiskLimitExceeded {
                risk_type: "volatility".to_string(),
                value: data.volatility,
                limit: self.config.max_volatility_threshold,
                account_id: "system".to_string(),
                limit_type: RiskLimitType::Volatility,
            });
        }
        
        // Check for volatility spikes
        if let Some(historical_vol) = self.get_historical_volatility(&data.symbol) {
            let vol_ratio = data.volatility / historical_vol;
            
            if vol_ratio > self.config.volatility_spike_threshold {
                return Err(TalebianSecureError::MarketRegimeChange {
                    details: format!("Volatility spike detected: {:.1}x historical average", vol_ratio),
                    previous_regime: "normal_volatility".to_string(),
                    current_regime: "high_volatility".to_string(),
                    confidence: 0.9,
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate liquidity requirements
    fn validate_liquidity_requirements(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // Check minimum volume requirements
        if data.volume < self.config.min_volume_threshold {
            return Err(TalebianSecureError::InsufficientLiquidity {
                required: self.config.min_volume_threshold,
                available: data.volume,
                market_symbol: data.symbol.clone().unwrap_or_default(),
                side: TradingSide::Buy, // Default side
            });
        }
        
        // Check bid-ask spread for liquidity
        let spread = data.ask - data.bid;
        let spread_bps = (spread / data.price) * 10000.0; // basis points
        
        if spread_bps > self.config.max_spread_bps {
            return Err(TalebianSecureError::InsufficientLiquidity {
                required: self.config.max_spread_bps / 10000.0 * data.price,
                available: spread,
                market_symbol: data.symbol.clone().unwrap_or_default(),
                side: TradingSide::Buy,
            });
        }
        
        Ok(())
    }
    
    /// Validate risk parameters are within acceptable bounds
    fn validate_risk_parameters(&self, data: &MarketData) -> Result<(), TalebianSecureError> {
        // This would typically validate against account-specific or system-wide risk limits
        // For now, we'll validate general risk indicators
        
        // Check for data quality risk
        if self.calculate_data_quality_score(data) < self.config.min_data_quality_score {
            return Err(TalebianSecureError::DataIntegrity {
                details: "Data quality score below minimum threshold".to_string(),
                data_hash: None,
                expected_hash: None,
            });
        }
        
        Ok(())
    }
    
    /// Calculate data quality score
    fn calculate_data_quality_score(&self, data: &MarketData) -> f64 {
        let mut score = 1.0;
        
        // Reduce score for missing optional data
        if data.bid_volume.is_none() { score *= 0.9; }
        if data.ask_volume.is_none() { score *= 0.9; }
        if data.returns.is_empty() { score *= 0.8; }
        if data.volume_history.is_empty() { score *= 0.8; }
        
        // Reduce score for unusual values
        let spread_pct = (data.ask - data.bid) / data.price;
        if spread_pct > 0.01 { score *= 0.9; } // Wide spread
        
        score.max(0.0).min(1.0)
    }
    
    /// Get previous price for comparison (stub implementation)
    fn get_previous_price(&self, _symbol: &str) -> Option<f64> {
        // Implementation would fetch from historical data store
        None
    }
    
    /// Get historical volatility (stub implementation)
    fn get_historical_volatility(&self, _symbol: &Option<String>) -> Option<f64> {
        // Implementation would calculate from historical data
        None
    }
    
    /// Validate after-hours data with different rules
    fn validate_after_hours_data(
        &self, 
        data: &MarketData, 
        _market_time: &chrono::DateTime<chrono::Utc>
    ) -> Result<(), TalebianSecureError> {
        // After-hours data typically has lower volume and wider spreads
        if data.volume > self.config.max_after_hours_volume {
            return Err(TalebianSecureError::VolumeInvalid {
                volume: data.volume,
                reason: "Volume too high for after-hours trading".to_string(),
                market_symbol: data.symbol.clone().unwrap_or_default(),
                historical_avg: Some(self.config.typical_after_hours_volume),
            });
        }
        
        Ok(())
    }
}

/// Configuration for business logic validation
#[derive(Debug, Clone)]
pub struct BusinessValidationConfig {
    pub allow_after_hours_data: bool,
    pub market_crash_threshold: f64,
    pub max_price_movement_threshold: f64,
    pub min_volatility_threshold: f64,
    pub max_volatility_threshold: f64,
    pub volatility_spike_threshold: f64,
    pub min_volume_threshold: f64,
    pub max_spread_bps: f64,
    pub min_data_quality_score: f64,
    pub max_after_hours_volume: f64,
    pub typical_after_hours_volume: f64,
}

impl Default for BusinessValidationConfig {
    fn default() -> Self {
        Self {
            allow_after_hours_data: true,
            market_crash_threshold: 0.1,  // 10% crash threshold
            max_price_movement_threshold: 0.2,  // 20% max price movement
            min_volatility_threshold: 0.0001,  // 0.01% minimum volatility
            max_volatility_threshold: 2.0,  // 200% maximum volatility
            volatility_spike_threshold: 3.0,  // 3x historical volatility
            min_volume_threshold: 1.0,  // Minimum volume
            max_spread_bps: 100.0,  // 100 basis points max spread
            min_data_quality_score: 0.7,  // 70% minimum quality
            max_after_hours_volume: 10000.0,  // Max after-hours volume
            typical_after_hours_volume: 1000.0,  // Typical after-hours volume
        }
    }
}
```

This input validation pipeline provides comprehensive protection against malicious inputs, data corruption, and business rule violations through a systematic multi-layer approach. Each layer builds upon the previous one, ensuring that only valid, consistent, and business-appropriate data reaches the core trading algorithms.