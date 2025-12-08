//! Input Validation and Injection Attack Prevention
//! 
//! Provides comprehensive input validation, sanitization, and injection attack
//! prevention for neural network trading systems.

use std::collections::HashMap;
use anyhow::{Result, anyhow};
use regex::Regex;
use serde::{Serialize, Deserialize};
use tracing::{debug, warn, error};

/// Input validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable strict validation mode
    pub strict_mode: bool,
    /// Maximum input length
    pub max_input_length: usize,
    /// Allow special characters
    pub allow_special_chars: bool,
    /// Enable SQL injection detection
    pub sql_injection_detection: bool,
    /// Enable XSS prevention
    pub xss_prevention: bool,
    /// Custom validation patterns
    pub custom_patterns: HashMap<String, String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        let mut custom_patterns = HashMap::new();
        custom_patterns.insert("trading_symbol".to_string(), r"^[A-Z]{1,8}$".to_string());
        custom_patterns.insert("price".to_string(), r"^\d+\.?\d*$".to_string());
        custom_patterns.insert("volume".to_string(), r"^\d+\.?\d*$".to_string());
        
        Self {
            strict_mode: true,
            max_input_length: 10000,
            allow_special_chars: false,
            sql_injection_detection: true,
            xss_prevention: true,
            custom_patterns,
        }
    }
}

/// Input validator with comprehensive security checks
pub struct InputValidator {
    config: ValidationConfig,
    sql_injection_patterns: Vec<Regex>,
    xss_patterns: Vec<Regex>,
    custom_validators: HashMap<String, Regex>,
}

impl InputValidator {
    /// Create new input validator
    pub fn new(config: ValidationConfig) -> Result<Self> {
        let sql_injection_patterns = Self::build_sql_injection_patterns()?;
        let xss_patterns = Self::build_xss_patterns()?;
        let custom_validators = Self::build_custom_validators(&config)?;

        Ok(Self {
            config,
            sql_injection_patterns,
            xss_patterns,
            custom_validators,
        })
    }

    /// Validate trading input data
    pub fn validate_trading_input(&self, input: &TradingInput) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();

        // Validate symbol
        if let Err(e) = self.validate_symbol(&input.symbol) {
            result.add_error(format!("Symbol validation failed: {}", e));
        }

        // Validate price
        if let Err(e) = self.validate_price(input.price) {
            result.add_error(format!("Price validation failed: {}", e));
        }

        // Validate volume
        if let Err(e) = self.validate_volume(input.volume) {
            result.add_error(format!("Volume validation failed: {}", e));
        }

        // Validate order type
        if let Err(e) = self.validate_order_type(&input.order_type) {
            result.add_error(format!("Order type validation failed: {}", e));
        }

        // Check for injection attacks
        if let Err(e) = self.check_injection_attacks(&input.to_string()) {
            result.add_error(format!("Security validation failed: {}", e));
        }

        result.is_valid = result.errors.is_empty();
        Ok(result)
    }

    /// Validate neural network model input
    pub fn validate_model_input(&self, input: &ModelInput) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();

        // Validate input dimensions
        if input.features.is_empty() {
            result.add_error("Input features cannot be empty".to_string());
        }

        if input.features.len() > 10000 {
            result.add_error("Input features exceed maximum allowed size".to_string());
        }

        // Validate feature values
        for (i, &value) in input.features.iter().enumerate() {
            if !value.is_finite() {
                result.add_error(format!("Feature {} contains invalid value: {}", i, value));
            }
            
            if value.abs() > 1e6 {
                result.add_warning(format!("Feature {} has unusually large value: {}", i, value));
            }
        }

        // Validate metadata
        if let Some(metadata) = &input.metadata {
            if let Err(e) = self.validate_string_field(metadata, "metadata") {
                result.add_error(format!("Metadata validation failed: {}", e));
            }
        }

        result.is_valid = result.errors.is_empty();
        Ok(result)
    }

    /// Validate user input data
    pub fn validate_user_input(&self, input: &UserInput) -> Result<ValidationResult> {
        let mut result = ValidationResult::new();

        // Validate username
        if let Err(e) = self.validate_username(&input.username) {
            result.add_error(format!("Username validation failed: {}", e));
        }

        // Validate password
        if let Err(e) = self.validate_password(&input.password) {
            result.add_error(format!("Password validation failed: {}", e));
        }

        // Validate email if provided
        if let Some(email) = &input.email {
            if let Err(e) = self.validate_email(email) {
                result.add_error(format!("Email validation failed: {}", e));
            }
        }

        // Check for injection attacks in all string fields
        for field in &[&input.username, &input.password] {
            if let Err(e) = self.check_injection_attacks(field) {
                result.add_error(format!("Security validation failed: {}", e));
            }
        }

        result.is_valid = result.errors.is_empty();
        Ok(result)
    }

    /// Sanitize input string to prevent attacks
    pub fn sanitize_input(&self, input: &str) -> String {
        let mut sanitized = input.to_string();

        // Remove common injection patterns
        if self.config.sql_injection_detection {
            sanitized = self.remove_sql_injection_patterns(&sanitized);
        }

        if self.config.xss_prevention {
            sanitized = self.remove_xss_patterns(&sanitized);
        }

        // Limit length
        if sanitized.len() > self.config.max_input_length {
            sanitized.truncate(self.config.max_input_length);
        }

        // Remove special characters if not allowed
        if !self.config.allow_special_chars {
            sanitized = sanitized.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '.' || *c == '-' || *c == '_')
                .collect();
        }

        sanitized
    }

    /// Validate trading symbol
    fn validate_symbol(&self, symbol: &str) -> Result<()> {
        if symbol.is_empty() {
            return Err(anyhow!("Symbol cannot be empty"));
        }

        if symbol.len() > 10 {
            return Err(anyhow!("Symbol too long"));
        }

        if !symbol.chars().all(|c| c.is_ascii_alphabetic()) {
            return Err(anyhow!("Symbol must contain only letters"));
        }

        // Check against custom pattern if defined
        if let Some(validator) = self.custom_validators.get("trading_symbol") {
            if !validator.is_match(symbol) {
                return Err(anyhow!("Symbol does not match required pattern"));
            }
        }

        Ok(())
    }

    /// Validate price value
    fn validate_price(&self, price: f64) -> Result<()> {
        if !price.is_finite() {
            return Err(anyhow!("Price must be a finite number"));
        }

        if price <= 0.0 {
            return Err(anyhow!("Price must be positive"));
        }

        if price > 1_000_000.0 {
            return Err(anyhow!("Price exceeds maximum allowed value"));
        }

        Ok(())
    }

    /// Validate volume value
    fn validate_volume(&self, volume: f64) -> Result<()> {
        if !volume.is_finite() {
            return Err(anyhow!("Volume must be a finite number"));
        }

        if volume <= 0.0 {
            return Err(anyhow!("Volume must be positive"));
        }

        if volume > 10_000_000.0 {
            return Err(anyhow!("Volume exceeds maximum allowed value"));
        }

        Ok(())
    }

    /// Validate order type
    fn validate_order_type(&self, order_type: &str) -> Result<()> {
        let valid_types = ["market", "limit", "stop", "stop_limit"];
        
        if !valid_types.contains(&order_type.to_lowercase().as_str()) {
            return Err(anyhow!("Invalid order type"));
        }

        Ok(())
    }

    /// Validate username
    fn validate_username(&self, username: &str) -> Result<()> {
        if username.is_empty() {
            return Err(anyhow!("Username cannot be empty"));
        }

        if username.len() < 3 || username.len() > 50 {
            return Err(anyhow!("Username must be between 3 and 50 characters"));
        }

        if !username.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err(anyhow!("Username contains invalid characters"));
        }

        Ok(())
    }

    /// Validate password
    fn validate_password(&self, password: &str) -> Result<()> {
        if password.len() < 8 {
            return Err(anyhow!("Password must be at least 8 characters"));
        }

        if password.len() > 128 {
            return Err(anyhow!("Password too long"));
        }

        let has_upper = password.chars().any(|c| c.is_uppercase());
        let has_lower = password.chars().any(|c| c.is_lowercase());
        let has_digit = password.chars().any(|c| c.is_numeric());
        let has_special = password.chars().any(|c| !c.is_alphanumeric());

        if !has_upper || !has_lower || !has_digit || !has_special {
            return Err(anyhow!("Password must contain uppercase, lowercase, digit, and special character"));
        }

        Ok(())
    }

    /// Validate email address
    fn validate_email(&self, email: &str) -> Result<()> {
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")?;
        
        if !email_regex.is_match(email) {
            return Err(anyhow!("Invalid email format"));
        }

        Ok(())
    }

    /// Validate generic string field
    fn validate_string_field(&self, value: &str, field_name: &str) -> Result<()> {
        if value.len() > self.config.max_input_length {
            return Err(anyhow!("{} exceeds maximum length", field_name));
        }

        // Check for null bytes
        if value.contains('\0') {
            return Err(anyhow!("{} contains null bytes", field_name));
        }

        // Check for control characters
        if value.chars().any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t') {
            return Err(anyhow!("{} contains invalid control characters", field_name));
        }

        Ok(())
    }

    /// Check for SQL injection attacks
    fn check_injection_attacks(&self, input: &str) -> Result<()> {
        if self.config.sql_injection_detection {
            for pattern in &self.sql_injection_patterns {
                if pattern.is_match(input) {
                    return Err(anyhow!("Potential SQL injection detected"));
                }
            }
        }

        if self.config.xss_prevention {
            for pattern in &self.xss_patterns {
                if pattern.is_match(input) {
                    return Err(anyhow!("Potential XSS attack detected"));
                }
            }
        }

        Ok(())
    }

    /// Remove SQL injection patterns
    fn remove_sql_injection_patterns(&self, input: &str) -> String {
        let mut sanitized = input.to_string();
        
        for pattern in &self.sql_injection_patterns {
            sanitized = pattern.replace_all(&sanitized, "").to_string();
        }
        
        sanitized
    }

    /// Remove XSS patterns
    fn remove_xss_patterns(&self, input: &str) -> String {
        let mut sanitized = input.to_string();
        
        for pattern in &self.xss_patterns {
            sanitized = pattern.replace_all(&sanitized, "").to_string();
        }
        
        sanitized
    }

    /// Build SQL injection detection patterns
    fn build_sql_injection_patterns() -> Result<Vec<Regex>> {
        let patterns = vec![
            r"(?i)(\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b)",
            r"(?i)(\b(or|and)\s+\d+\s*=\s*\d+)",
            r"(?i)(\'\s*(or|and)\s*\'\w*\'\s*=\s*\'\w*\')",
            r"(?i)(\/\*|\*\/|--|\#)",
            r"(?i)(\bxp_cmdshell\b|\bsp_executesql\b)",
            r"(?i)(\b(cast|convert|char|varchar)\s*\()",
        ];

        patterns.into_iter()
            .map(|p| Regex::new(p).map_err(|e| anyhow!("Failed to compile regex: {}", e)))
            .collect()
    }

    /// Build XSS detection patterns
    fn build_xss_patterns() -> Result<Vec<Regex>> {
        let patterns = vec![
            r"(?i)(<script[\s\S]*?>[\s\S]*?</script>)",
            r"(?i)(<iframe[\s\S]*?>[\s\S]*?</iframe>)",
            r"(?i)(javascript\s*:)",
            r"(?i)(on\w+\s*=)",
            r"(?i)(<img[\s\S]*?onerror[\s\S]*?>)",
            r"(?i)(<svg[\s\S]*?onload[\s\S]*?>)",
        ];

        patterns.into_iter()
            .map(|p| Regex::new(p).map_err(|e| anyhow!("Failed to compile regex: {}", e)))
            .collect()
    }

    /// Build custom validators from config
    fn build_custom_validators(config: &ValidationConfig) -> Result<HashMap<String, Regex>> {
        let mut validators = HashMap::new();
        
        for (name, pattern) in &config.custom_patterns {
            let regex = Regex::new(pattern)
                .map_err(|e| anyhow!("Failed to compile custom pattern '{}': {}", name, e))?;
            validators.insert(name.clone(), regex);
        }
        
        Ok(validators)
    }
}

/// Input sanitizer for safe data processing
pub struct InputSanitizer {
    validator: InputValidator,
}

impl InputSanitizer {
    /// Create new input sanitizer
    pub fn new(config: ValidationConfig) -> Result<Self> {
        let validator = InputValidator::new(config)?;
        Ok(Self { validator })
    }

    /// Sanitize and validate trading input
    pub fn sanitize_trading_input(&self, input: TradingInput) -> Result<SanitizedTradingInput> {
        // Validate first
        let validation = self.validator.validate_trading_input(&input)?;
        if !validation.is_valid {
            return Err(anyhow!("Validation failed: {:?}", validation.errors));
        }

        // Sanitize string fields
        let sanitized_symbol = self.validator.sanitize_input(&input.symbol);
        let sanitized_order_type = self.validator.sanitize_input(&input.order_type);

        Ok(SanitizedTradingInput {
            symbol: sanitized_symbol,
            price: input.price,
            volume: input.volume,
            order_type: sanitized_order_type,
            timestamp: input.timestamp,
            validation_metadata: validation,
        })
    }

    /// Sanitize and validate model input
    pub fn sanitize_model_input(&self, input: ModelInput) -> Result<SanitizedModelInput> {
        // Validate first
        let validation = self.validator.validate_model_input(&input)?;
        if !validation.is_valid {
            return Err(anyhow!("Validation failed: {:?}", validation.errors));
        }

        // Sanitize metadata if present
        let sanitized_metadata = input.metadata
            .map(|m| self.validator.sanitize_input(&m));

        Ok(SanitizedModelInput {
            features: input.features,
            metadata: sanitized_metadata,
            validation_metadata: validation,
        })
    }

    /// Sanitize and validate user input
    pub fn sanitize_user_input(&self, input: UserInput) -> Result<SanitizedUserInput> {
        // Validate first
        let validation = self.validator.validate_user_input(&input)?;
        if !validation.is_valid {
            return Err(anyhow!("Validation failed: {:?}", validation.errors));
        }

        // Sanitize string fields
        let sanitized_username = self.validator.sanitize_input(&input.username);
        let sanitized_email = input.email
            .map(|e| self.validator.sanitize_input(&e));

        Ok(SanitizedUserInput {
            username: sanitized_username,
            password: input.password, // Don't sanitize passwords, just validate
            email: sanitized_email,
            validation_metadata: validation,
        })
    }
}

// Data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingInput {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub order_type: String,
    pub timestamp: u64,
}

impl ToString for TradingInput {
    fn to_string(&self) -> String {
        format!("{} {} {} {} {}", self.symbol, self.price, self.volume, self.order_type, self.timestamp)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInput {
    pub features: Vec<f64>,
    pub metadata: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInput {
    pub username: String,
    pub password: String,
    pub email: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedTradingInput {
    pub symbol: String,
    pub price: f64,
    pub volume: f64,
    pub order_type: String,
    pub timestamp: u64,
    pub validation_metadata: ValidationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedModelInput {
    pub features: Vec<f64>,
    pub metadata: Option<String>,
    pub validation_metadata: ValidationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedUserInput {
    pub username: String,
    pub password: String,
    pub email: Option<String>,
    pub validation_metadata: ValidationResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn new() -> Self {
        Self {
            is_valid: false,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trading_input_validation() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();

        let valid_input = TradingInput {
            symbol: "AAPL".to_string(),
            price: 150.0,
            volume: 1000.0,
            order_type: "market".to_string(),
            timestamp: 1234567890,
        };

        let result = validator.validate_trading_input(&valid_input).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_sql_injection_detection() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();

        let malicious_input = TradingInput {
            symbol: "AAPL'; DROP TABLE orders; --".to_string(),
            price: 150.0,
            volume: 1000.0,
            order_type: "market".to_string(),
            timestamp: 1234567890,
        };

        let result = validator.validate_trading_input(&malicious_input).unwrap();
        assert!(!result.is_valid);
    }

    #[test]
    fn test_input_sanitization() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();

        let malicious_input = "<script>alert('xss')</script>test";
        let sanitized = validator.sanitize_input(malicious_input);
        
        assert!(!sanitized.contains("<script>"));
        assert!(!sanitized.contains("alert"));
    }

    #[test]
    fn test_model_input_validation() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();

        let valid_input = ModelInput {
            features: vec![1.0, 2.0, 3.0],
            metadata: Some("test metadata".to_string()),
        };

        let result = validator.validate_model_input(&valid_input).unwrap();
        assert!(result.is_valid);
    }

    #[test]
    fn test_password_validation() {
        let config = ValidationConfig::default();
        let validator = InputValidator::new(config).unwrap();

        // Valid password
        assert!(validator.validate_password("SecurePass123!").is_ok());

        // Invalid passwords
        assert!(validator.validate_password("weak").is_err()); // Too short
        assert!(validator.validate_password("nouppercase123!").is_err()); // No uppercase
        assert!(validator.validate_password("NOLOWERCASE123!").is_err()); // No lowercase
        assert!(validator.validate_password("NoDigits!").is_err()); // No digits
        assert!(validator.validate_password("NoSpecial123").is_err()); // No special chars
    }
}