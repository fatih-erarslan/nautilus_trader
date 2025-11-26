//! XSS (Cross-Site Scripting) Protection Module
//!
//! Provides comprehensive protection against XSS attacks through:
//! - Pattern-based detection of common XSS vectors
//! - HTML entity validation
//! - Safe HTML escaping
//! - Context-aware validation

use anyhow::{anyhow, Result};
use regex::Regex;
use std::sync::OnceLock;

/// XSS patterns to detect in user input
static XSS_PATTERNS: &[&str] = &[
    // Script injection
    "<script",
    "</script",
    "javascript:",
    "data:text/html",

    // Event handlers
    "onerror=",
    "onload=",
    "onclick=",
    "onmouseover=",
    "onmouseenter=",
    "onmouseleave=",
    "onfocus=",
    "onblur=",
    "onchange=",
    "onsubmit=",
    "onkeypress=",
    "onkeydown=",
    "onkeyup=",

    // Dangerous tags
    "<iframe",
    "<embed",
    "<object",
    "<applet",
    "<meta",
    "<link",
    "<style",
    "<form",

    // JavaScript functions
    "eval(",
    "expression(",
    "fromcharcode",
    "document.cookie",
    "document.write",
    "window.location",

    // Protocol handlers
    "vbscript:",
    "livescript:",
    "mocha:",
    "feed:",
    "data:text",

    // SVG vectors
    "<svg",
    "<foreignobject",
    "<animate",
    "<set",

    // Import/require
    "@import",
    "import(",
    "require(",
];

/// Additional regex patterns for complex XSS detection
fn xss_regex_patterns() -> &'static Vec<Regex> {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            // Hex/unicode encoded javascript
            Regex::new(r"(?i)\\x6a\\x61\\x76\\x61\\x73\\x63\\x72\\x69\\x70\\x74").unwrap(),
            // Base64 encoded patterns
            Regex::new(r"(?i)base64\s*,").unwrap(),
            // Data URLs with scripts
            Regex::new(r"(?i)data:[\w/]+;base64").unwrap(),
            // Expression in CSS
            Regex::new(r"(?i)expression\s*\(").unwrap(),
            // Import with URL
            Regex::new(r#"(?i)@import\s+['"]"#).unwrap(),
            // Multiple spaces (obfuscation)
            Regex::new(r"<\s*script").unwrap(),
        ]
    })
}

/// Validate that a string does not contain XSS patterns
///
/// # Arguments
/// * `value` - The string to validate
/// * `field_name` - Name of the field for error messages
///
/// # Returns
/// * `Ok(())` if validation passes
/// * `Err` if potential XSS is detected
///
/// # Example
/// ```
/// use neural_trader_backend::security::xss_protection::validate_no_xss;
///
/// assert!(validate_no_xss("Hello World", "message").is_ok());
/// assert!(validate_no_xss("<script>alert(1)</script>", "message").is_err());
/// ```
pub fn validate_no_xss(value: &str, field_name: &str) -> Result<()> {
    if value.is_empty() {
        return Ok(());
    }

    let value_lower = value.to_lowercase();

    // Check simple string patterns
    for pattern in XSS_PATTERNS {
        if value_lower.contains(pattern) {
            return Err(anyhow!(
                "Potential XSS detected in '{}': contains '{}'",
                field_name,
                pattern
            ));
        }
    }

    // Check regex patterns
    for regex in xss_regex_patterns() {
        if regex.is_match(value) {
            return Err(anyhow!(
                "Potential XSS detected in '{}': matches suspicious pattern",
                field_name
            ));
        }
    }

    // Check for HTML entities that could be used for obfuscation
    if value.contains("&#") {
        // Allow only common safe entities
        let safe_entities = ["&#39;", "&#34;", "&#38;", "&#60;", "&#62;"];
        let has_unsafe = value
            .split("&#")
            .skip(1)
            .any(|part| {
                let entity = format!("&#{}", part.chars().take(10).collect::<String>());
                !safe_entities.iter().any(|safe| entity.starts_with(safe))
            });

        if has_unsafe {
            return Err(anyhow!(
                "Suspicious HTML entities detected in '{}'",
                field_name
            ));
        }
    }

    // Check for encoded characters
    if value.contains("&lt;") || value.contains("&gt;") {
        return Err(anyhow!(
            "Pre-encoded HTML not allowed in '{}'",
            field_name
        ));
    }

    // Check for null bytes (can bypass some validators)
    if value.contains('\0') {
        return Err(anyhow!(
            "Null bytes not allowed in '{}'",
            field_name
        ));
    }

    Ok(())
}

/// Escape HTML special characters to prevent XSS
///
/// Converts potentially dangerous characters to their HTML entity equivalents:
/// - `&` - `&amp;`
/// - `<` - `&lt;`
/// - `>` - `&gt;`
/// - `"` - `&quot;`
/// - `'` - `&#x27;`
/// - `/` - `&#x2F;`
///
/// # Example
/// ```
/// use neural_trader_backend::security::xss_protection::escape_html;
///
/// assert_eq!(escape_html("<script>alert(1)</script>"), "&lt;script&gt;alert(1)&lt;&#x2F;script&gt;");
/// ```
pub fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
        .replace('/', "&#x2F;")
}

/// Validate and escape user input in one step
///
/// First validates for XSS patterns, then escapes HTML if validation passes.
///
/// # Example
/// ```
/// use neural_trader_backend::security::xss_protection::validate_and_escape;
///
/// assert!(validate_and_escape("Hello <World>", "name").is_ok());
/// assert!(validate_and_escape("<script>alert(1)</script>", "name").is_err());
/// ```
pub fn validate_and_escape(value: &str, field_name: &str) -> Result<String> {
    validate_no_xss(value, field_name)?;
    Ok(escape_html(value))
}

/// Context-aware validation for specific input types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InputContext {
    /// Plain text (most restrictive)
    PlainText,
    /// Email addresses
    Email,
    /// URLs (allows more characters but validates format)
    Url,
    /// Markdown content (allows some formatting)
    Markdown,
}

/// Validate input based on context
pub fn validate_input_context(value: &str, field_name: &str, context: InputContext) -> Result<()> {
    // Always check for basic XSS
    validate_no_xss(value, field_name)?;

    match context {
        InputContext::PlainText => {
            // No additional validation needed
            Ok(())
        }
        InputContext::Email => {
            // Basic email format validation
            if !value.contains('@') || value.contains('<') || value.contains('>') {
                return Err(anyhow!("Invalid email format in '{}'", field_name));
            }
            Ok(())
        }
        InputContext::Url => {
            // Basic URL validation
            if !value.starts_with("http://") && !value.starts_with("https://") {
                return Err(anyhow!("Invalid URL format in '{}'", field_name));
            }
            // Check for javascript: protocol (should be caught by XSS check but extra safety)
            if value.to_lowercase().contains("javascript:") {
                return Err(anyhow!("JavaScript URLs not allowed in '{}'", field_name));
            }
            Ok(())
        }
        InputContext::Markdown => {
            // Markdown allows some HTML-like syntax, but we still block dangerous patterns
            // The XSS check already covers this
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_no_xss_safe_input() {
        assert!(validate_no_xss("Hello World", "message").is_ok());
        assert!(validate_no_xss("Test-123_ABC", "name").is_ok());
        assert!(validate_no_xss("user@example.com", "email").is_ok());
    }

    #[test]
    fn test_validate_no_xss_script_tags() {
        assert!(validate_no_xss("<script>alert(1)</script>", "message").is_err());
        assert!(validate_no_xss("Hello<script>alert(1)", "message").is_err());
        assert!(validate_no_xss("<SCRIPT>alert(1)</SCRIPT>", "message").is_err());
    }

    #[test]
    fn test_validate_no_xss_event_handlers() {
        assert!(validate_no_xss("onclick=alert(1)", "message").is_err());
        assert!(validate_no_xss("onerror=alert(1)", "message").is_err());
        assert!(validate_no_xss("onload=malicious()", "message").is_err());
    }

    #[test]
    fn test_validate_no_xss_javascript_protocol() {
        assert!(validate_no_xss("javascript:alert(1)", "url").is_err());
        assert!(validate_no_xss("JavaScript:void(0)", "url").is_err());
    }

    #[test]
    fn test_validate_no_xss_dangerous_tags() {
        assert!(validate_no_xss("<iframe src='evil.com'>", "content").is_err());
        assert!(validate_no_xss("<embed src='evil.swf'>", "content").is_err());
        assert!(validate_no_xss("<object data='evil'>", "content").is_err());
    }

    #[test]
    fn test_escape_html() {
        assert_eq!(escape_html("Hello World"), "Hello World");
        assert_eq!(
            escape_html("<script>alert(1)</script>"),
            "&lt;script&gt;alert(1)&lt;&#x2F;script&gt;"
        );
        assert_eq!(escape_html("A&B"), "A&amp;B");
        assert_eq!(escape_html("'quote'"), "&#x27;quote&#x27;");
    }

    #[test]
    fn test_validate_and_escape() {
        assert!(validate_and_escape("Hello <World>", "name").is_ok());
        assert!(validate_and_escape("<script>", "name").is_err());
    }

    #[test]
    fn test_context_validation_email() {
        assert!(validate_input_context(
            "user@example.com",
            "email",
            InputContext::Email
        ).is_ok());
        assert!(validate_input_context(
            "not-an-email",
            "email",
            InputContext::Email
        ).is_err());
    }

    #[test]
    fn test_context_validation_url() {
        assert!(validate_input_context(
            "https://example.com",
            "url",
            InputContext::Url
        ).is_ok());
        assert!(validate_input_context(
            "javascript:alert(1)",
            "url",
            InputContext::Url
        ).is_err());
    }
}
