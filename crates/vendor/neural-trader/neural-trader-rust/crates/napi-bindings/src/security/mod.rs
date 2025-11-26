//! Security Module
//!
//! Comprehensive security utilities for protecting against common web vulnerabilities:
//! - XSS (Cross-Site Scripting) protection
//! - Path traversal prevention
//! - Input validation and sanitization
//! - SQL injection protection (via parameterized queries)
//! - Rate limiting and DoS protection

pub mod xss_protection;
pub mod path_validation;

// Re-export commonly used functions
pub use xss_protection::{
    validate_no_xss,
    escape_html,
    validate_and_escape,
    validate_input_context,
    InputContext,
};

pub use path_validation::{
    validate_safe_path,
    validate_filename,
    sanitize_filename,
    get_safe_extension,
    validate_extension,
    allowlists,
};

/// Security configuration for the application
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Maximum length for user input fields
    pub max_input_length: usize,
    /// Maximum length for filenames
    pub max_filename_length: usize,
    /// Allowed file extensions for uploads
    pub allowed_extensions: Vec<String>,
    /// Enable strict XSS validation
    pub strict_xss_validation: bool,
    /// Enable path validation
    pub enable_path_validation: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            max_input_length: 10_000,
            max_filename_length: 255,
            allowed_extensions: vec![
                "txt".to_string(),
                "json".to_string(),
                "csv".to_string(),
                "md".to_string(),
            ],
            strict_xss_validation: true,
            enable_path_validation: true,
        }
    }
}

/// Validate user input with comprehensive security checks
///
/// Applies multiple security validations:
/// - Length limits
/// - XSS pattern detection
/// - Context-aware validation
///
/// # Example
/// ```
/// use neural_trader_backend::security::{validate_user_input, InputContext};
///
/// let input = "user@example.com";
/// assert!(validate_user_input(input, "email", InputContext::Email, 100).is_ok());
/// ```
pub fn validate_user_input(
    value: &str,
    field_name: &str,
    context: InputContext,
    max_length: usize,
) -> anyhow::Result<()> {
    // Check length
    if value.len() > max_length {
        return Err(anyhow::anyhow!(
            "Input '{}' exceeds maximum length of {} characters (got {})",
            field_name,
            max_length,
            value.len()
        ));
    }

    // Validate for XSS
    validate_no_xss(value, field_name)?;

    // Context-specific validation
    validate_input_context(value, field_name, context)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_config_default() {
        let config = SecurityConfig::default();
        assert_eq!(config.max_input_length, 10_000);
        assert_eq!(config.max_filename_length, 255);
        assert!(config.strict_xss_validation);
        assert!(config.enable_path_validation);
    }

    #[test]
    fn test_validate_user_input() {
        assert!(validate_user_input(
            "Hello World",
            "message",
            InputContext::PlainText,
            100
        ).is_ok());

        // Test length limit
        let long_input = "a".repeat(1000);
        assert!(validate_user_input(
            &long_input,
            "message",
            InputContext::PlainText,
            100
        ).is_err());

        // Test XSS
        assert!(validate_user_input(
            "<script>alert(1)</script>",
            "message",
            InputContext::PlainText,
            100
        ).is_err());
    }
}
