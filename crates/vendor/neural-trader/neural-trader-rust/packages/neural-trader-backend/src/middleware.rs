//! Security Middleware Module
//!
//! Provides request validation, sanitization, and security checks:
//! - Input validation and sanitization
//! - SQL injection prevention
//! - XSS prevention
//! - Path traversal prevention
//! - Request validation
//! - CORS handling

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Request context with security information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct SecurityContext {
    pub user_id: Option<String>,
    pub username: Option<String>,
    pub role: String,
    pub ip_address: String,
    pub user_agent: Option<String>,
    pub request_id: String,
    pub timestamp: String,
}

/// Sanitization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizedInput {
    pub original: String,
    pub sanitized: String,
    pub threats_detected: Vec<String>,
    pub is_safe: bool,
}

/// Input sanitizer
pub struct InputSanitizer;

impl InputSanitizer {
    /// Sanitize string input to prevent SQL injection
    ///
    /// **SECURITY WARNING**: This is a defense-in-depth measure only.
    /// **ALWAYS** use parameterized queries or prepared statements at the database layer.
    /// This function cannot prevent all SQL injection attacks and should not be relied upon alone.
    ///
    /// # Security Strategy
    /// - This function **rejects** input containing suspicious patterns rather than sanitizing
    /// - It detects both plain and encoded SQL injection attempts (URL-encoded, hex-encoded, HTML entities)
    /// - Input is marked as unsafe if ANY threat is detected
    /// - Original input is preserved for logging and analysis
    ///
    /// # Best Practices
    /// 1. Use ORM or query builders with parameterized queries
    /// 2. Use prepared statements with bound parameters
    /// 3. Apply principle of least privilege to database accounts
    /// 4. Log and monitor all rejected inputs
    /// 5. Implement rate limiting on endpoints accepting user input
    pub fn sanitize_sql(input: &str) -> SanitizedInput {
        let mut threats = Vec::new();
        let input_lower = input.to_lowercase();
        let input_upper = input.to_uppercase();

        // SQL injection patterns to detect
        let sql_patterns = [
            ("--", "SQL comment"),
            (";", "SQL statement separator"),
            ("'", "SQL string delimiter"),
            ("\"", "SQL string delimiter"),
            ("xp_", "SQL Server extended procedure"),
            ("sp_", "SQL Server stored procedure"),
            ("DROP ", "DROP statement"),
            ("DELETE ", "DELETE statement"),
            ("INSERT ", "INSERT statement"),
            ("UPDATE ", "UPDATE statement"),
            ("UNION ", "UNION statement"),
            ("SELECT ", "SELECT statement"),
            ("EXEC ", "EXEC statement"),
            ("EXECUTE ", "EXECUTE statement"),
            ("CREATE ", "CREATE statement"),
            ("ALTER ", "ALTER statement"),
            ("TRUNCATE ", "TRUNCATE statement"),
            ("MERGE ", "MERGE statement"),
            ("GRANT ", "GRANT statement"),
            ("REVOKE ", "REVOKE statement"),
        ];

        for (pattern, description) in sql_patterns.iter() {
            if input_upper.contains(&pattern.to_uppercase()) {
                threats.push(format!("Potential SQL injection: {}", description));
            }
        }

        // Also check for encoded SQL injection attempts
        let encoded_patterns = [
            ("%27", "URL-encoded single quote"),
            ("%22", "URL-encoded double quote"),
            ("%3B", "URL-encoded semicolon"),
            ("%2D%2D", "URL-encoded comment"),
            ("%2d%2d", "URL-encoded comment (lowercase)"),
            ("0x27", "Hex-encoded quote"),
            ("0x22", "Hex-encoded quote"),
            ("0X27", "Hex-encoded quote (uppercase)"),
            ("0X22", "Hex-encoded quote (uppercase)"),
            ("&#39;", "HTML entity quote"),
            ("&#34;", "HTML entity quote"),
            ("&#x27;", "HTML entity hex quote"),
            ("&#x22;", "HTML entity hex quote"),
            ("%20OR%20", "URL-encoded OR operator"),
            ("%20AND%20", "URL-encoded AND operator"),
            ("+OR+", "Plus-encoded OR operator"),
            ("+AND+", "Plus-encoded AND operator"),
            ("%60", "URL-encoded backtick"),
            ("\\x27", "Backslash-hex encoded quote"),
            ("\\x22", "Backslash-hex encoded quote"),
            ("%00", "URL-encoded null byte"),
            ("\\0", "Escaped null byte"),
            ("%5C", "URL-encoded backslash"),
            ("%25", "URL-encoded percent (double encoding)"),
        ];

        for (pattern, description) in encoded_patterns.iter() {
            if input_lower.contains(&pattern.to_lowercase()) {
                threats.push(format!("Potential SQL injection (encoded): {}", description));
            }
        }

        // Check for multiple encoding attempts (double/triple encoding)
        if input_lower.contains("%25") {
            threats.push("Potential double-encoded SQL injection attempt".to_string());
        }

        // If ANY threats detected, reject the input entirely
        // Do NOT attempt to sanitize - reject and log
        let sanitized = if threats.is_empty() {
            input.to_string()
        } else {
            // Return empty string to indicate rejection
            // Caller should handle this by rejecting the request
            String::new()
        };

        SanitizedInput {
            original: input.to_string(),
            sanitized,
            threats_detected: threats.clone(),
            is_safe: threats.is_empty(),
        }
    }

    /// Sanitize string input to prevent XSS
    pub fn sanitize_xss(input: &str) -> SanitizedInput {
        let mut threats = Vec::new();
        let mut sanitized = input.to_string();

        // XSS patterns to detect and sanitize
        let xss_patterns = [
            ("<script", "Script tag"),
            ("javascript:", "JavaScript protocol"),
            ("onerror=", "Event handler"),
            ("onclick=", "Event handler"),
            ("onload=", "Event handler"),
            ("<iframe", "Iframe tag"),
            ("<object", "Object tag"),
            ("<embed", "Embed tag"),
        ];

        for (pattern, description) in xss_patterns.iter() {
            if input.to_lowercase().contains(&pattern.to_lowercase()) {
                threats.push(format!("Potential XSS: {}", description));
                // HTML encode dangerous characters
                sanitized = sanitized
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace("\"", "&quot;")
                    .replace("'", "&#x27;")
                    .replace("/", "&#x2F;");
                break; // Once we detect XSS, sanitize the whole string
            }
        }

        SanitizedInput {
            original: input.to_string(),
            sanitized,
            threats_detected: threats.clone(),
            is_safe: threats.is_empty(),
        }
    }

    /// Sanitize file path to prevent path traversal
    pub fn sanitize_path(input: &str) -> SanitizedInput {
        let mut threats = Vec::new();
        let mut sanitized = input.to_string();

        // Path traversal patterns
        let path_patterns = [
            ("../", "Directory traversal"),
            ("..\\", "Directory traversal (Windows)"),
            ("/etc/", "System directory access"),
            ("/root/", "Root directory access"),
            ("c:\\", "Windows system drive"),
            ("/proc/", "Process directory access"),
            ("/sys/", "System directory access"),
        ];

        for (pattern, description) in path_patterns.iter() {
            if input.to_lowercase().contains(&pattern.to_lowercase()) {
                threats.push(format!("Potential path traversal: {}", description));
                // Remove dangerous path components
                sanitized = sanitized.replace(pattern, "");
            }
        }

        // Remove null bytes
        if input.contains('\0') {
            threats.push("Null byte injection".to_string());
            sanitized = sanitized.replace('\0', "");
        }

        SanitizedInput {
            original: input.to_string(),
            sanitized,
            threats_detected: threats.clone(),
            is_safe: threats.is_empty(),
        }
    }

    /// Comprehensive input sanitization
    pub fn sanitize_all(input: &str) -> SanitizedInput {
        // Apply all sanitization checks
        let sql_result = Self::sanitize_sql(input);
        let xss_result = Self::sanitize_xss(&sql_result.sanitized);
        let path_result = Self::sanitize_path(&xss_result.sanitized);

        let mut all_threats = Vec::new();
        all_threats.extend(sql_result.threats_detected);
        all_threats.extend(xss_result.threats_detected);
        all_threats.extend(path_result.threats_detected);

        SanitizedInput {
            original: input.to_string(),
            sanitized: path_result.sanitized,
            threats_detected: all_threats.clone(),
            is_safe: all_threats.is_empty(),
        }
    }

    /// Validate email format
    pub fn validate_email(email: &str) -> bool {
        let email_regex = regex::Regex::new(
            r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        ).unwrap();
        email_regex.is_match(email)
    }

    /// Validate numeric input within range
    pub fn validate_number(value: f64, min: f64, max: f64) -> std::result::Result<f64, String> {
        if value.is_nan() || value.is_infinite() {
            return Err("Invalid number".to_string());
        }

        if value < min {
            return Err(format!("Value {} is below minimum {}", value, min));
        }

        if value > max {
            return Err(format!("Value {} exceeds maximum {}", value, max));
        }

        Ok(value)
    }

    /// Validate array length
    pub fn validate_array_length<T>(arr: &[T], min: usize, max: usize) -> std::result::Result<(), String> {
        let len = arr.len();
        if len < min {
            return Err(format!("Array length {} is below minimum {}", len, min));
        }

        if len > max {
            return Err(format!("Array length {} exceeds maximum {}", len, max));
        }

        Ok(())
    }
}

/// Request validator
pub struct RequestValidator;

impl RequestValidator {
    /// Validate trading request parameters
    pub fn validate_trading_request(
        symbol: &str,
        quantity: f64,
        price: Option<f64>,
    ) -> std::result::Result<(), String> {
        // Validate symbol format
        if symbol.is_empty() || symbol.len() > 10 {
            return Err("Invalid symbol format".to_string());
        }

        // Check for suspicious characters in symbol
        let sanitized = InputSanitizer::sanitize_all(symbol);
        if !sanitized.is_safe {
            return Err(format!("Invalid symbol: {:?}", sanitized.threats_detected));
        }

        // Validate quantity
        InputSanitizer::validate_number(quantity, 0.0001, 1_000_000.0)?;

        // Validate price if provided
        if let Some(p) = price {
            InputSanitizer::validate_number(p, 0.01, 1_000_000.0)?;
        }

        Ok(())
    }

    /// Validate portfolio request parameters
    pub fn validate_portfolio_request(allocations: &HashMap<String, f64>) -> std::result::Result<(), String> {
        // Check array length
        if allocations.is_empty() {
            return Err("Portfolio must have at least one allocation".to_string());
        }

        if allocations.len() > 100 {
            return Err("Portfolio cannot exceed 100 allocations".to_string());
        }

        // Validate each allocation
        let mut total_allocation = 0.0;
        for (symbol, allocation) in allocations {
            // Sanitize symbol
            let sanitized = InputSanitizer::sanitize_all(symbol);
            if !sanitized.is_safe {
                return Err(format!("Invalid symbol {}: {:?}", symbol, sanitized.threats_detected));
            }

            // Validate allocation percentage
            InputSanitizer::validate_number(*allocation, 0.0, 100.0)?;
            total_allocation += allocation;
        }

        // Validate total allocation
        if (total_allocation - 100.0).abs() > 0.01 {
            return Err(format!("Total allocation must equal 100%, got {}", total_allocation));
        }

        Ok(())
    }

    /// Validate API key format
    pub fn validate_api_key_format(key: &str) -> std::result::Result<(), String> {
        if !key.starts_with("ntk_") {
            return Err("API key must start with 'ntk_'".to_string());
        }

        if key.len() < 36 {
            return Err("API key too short".to_string());
        }

        // Check for suspicious patterns
        let sanitized = InputSanitizer::sanitize_all(key);
        if !sanitized.is_safe {
            return Err("API key contains invalid characters".to_string());
        }

        Ok(())
    }

    /// Validate date range
    pub fn validate_date_range(start: &str, end: &str) -> std::result::Result<(), String> {
        use chrono::{DateTime, Utc};

        let start_date: DateTime<Utc> = start.parse()
            .map_err(|_| "Invalid start date format".to_string())?;

        let end_date: DateTime<Utc> = end.parse()
            .map_err(|_| "Invalid end date format".to_string())?;

        if start_date >= end_date {
            return Err("Start date must be before end date".to_string());
        }

        let now = Utc::now();
        if end_date > now {
            return Err("End date cannot be in the future".to_string());
        }

        let max_range = chrono::Duration::days(365);
        if end_date - start_date > max_range {
            return Err("Date range cannot exceed 365 days".to_string());
        }

        Ok(())
    }
}

/// NAPI functions for input sanitization and validation

#[napi]
pub fn sanitize_input(input: String) -> Result<String> {
    let result = InputSanitizer::sanitize_all(&input);

    if !result.is_safe {
        tracing::warn!("Input sanitization detected threats: {:?}", result.threats_detected);
    }

    Ok(serde_json::to_string(&result)
        .map_err(|e| Error::from_reason(format!("JSON error: {}", e)))?)
}

#[napi]
pub fn validate_trading_params(
    symbol: String,
    quantity: f64,
    price: Option<f64>,
) -> Result<bool> {
    match RequestValidator::validate_trading_request(&symbol, quantity, price) {
        Ok(_) => Ok(true),
        Err(e) => Err(Error::from_reason(e)),
    }
}

#[napi]
pub fn validate_email_format(email: String) -> Result<bool> {
    Ok(InputSanitizer::validate_email(&email))
}

#[napi]
pub fn validate_api_key_format(key: String) -> Result<bool> {
    match RequestValidator::validate_api_key_format(&key) {
        Ok(_) => Ok(true),
        Err(e) => Err(Error::from_reason(e)),
    }
}

#[napi]
pub fn check_security_threats(input: String) -> Result<Vec<String>> {
    let result = InputSanitizer::sanitize_all(&input);
    Ok(result.threats_detected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_injection_detection() {
        let malicious = "'; DROP TABLE users; --";
        let result = InputSanitizer::sanitize_sql(malicious);
        assert!(!result.is_safe);
        assert!(!result.threats_detected.is_empty());
    }

    #[test]
    fn test_xss_detection() {
        let malicious = "<script>alert('XSS')</script>";
        let result = InputSanitizer::sanitize_xss(malicious);
        assert!(!result.is_safe);
        assert!(result.sanitized.contains("&lt;"));
    }

    #[test]
    fn test_path_traversal_detection() {
        let malicious = "../../etc/passwd";
        let result = InputSanitizer::sanitize_path(malicious);
        assert!(!result.is_safe);
        assert!(!result.threats_detected.is_empty());
    }

    #[test]
    fn test_email_validation() {
        assert!(InputSanitizer::validate_email("test@example.com"));
        assert!(!InputSanitizer::validate_email("invalid-email"));
        assert!(!InputSanitizer::validate_email("test@"));
    }

    #[test]
    fn test_number_validation() {
        assert!(InputSanitizer::validate_number(50.0, 0.0, 100.0).is_ok());
        assert!(InputSanitizer::validate_number(-1.0, 0.0, 100.0).is_err());
        assert!(InputSanitizer::validate_number(101.0, 0.0, 100.0).is_err());
    }

    #[test]
    fn test_trading_request_validation() {
        assert!(RequestValidator::validate_trading_request("AAPL", 10.0, Some(150.0)).is_ok());
        assert!(RequestValidator::validate_trading_request("", 10.0, None).is_err());
        assert!(RequestValidator::validate_trading_request("AAPL", -10.0, None).is_err());
    }
}
