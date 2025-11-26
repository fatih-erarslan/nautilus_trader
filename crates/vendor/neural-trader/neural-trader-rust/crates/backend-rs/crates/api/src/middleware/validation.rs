use axum::{
    http::StatusCode,
    Json,
};
use serde_json::json;
use validator::Validate;

// Input validation utilities
pub fn validate_url(url: &str) -> Result<(), String> {
    if url.is_empty() {
        return Err("URL cannot be empty".to_string());
    }

    if url.len() > 2048 {
        return Err("URL is too long (max 2048 characters)".to_string());
    }

    // Check for basic URL structure
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err("URL must start with http:// or https://".to_string());
    }

    // Check for suspicious characters that might indicate injection attempts
    let suspicious_chars = ['<', '>', '"', '\'', '{', '}', '|', '\\', '^', '`'];
    if url.chars().any(|c| suspicious_chars.contains(&c)) {
        return Err("URL contains invalid characters".to_string());
    }

    Ok(())
}

pub fn validate_scan_type(scan_type: &str) -> Result<(), String> {
    match scan_type {
        "openapi" | "auto" | "rest" | "graphql" => Ok(()),
        _ => Err(format!("Invalid scan type: {}", scan_type)),
    }
}

pub fn validate_workflow_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("Workflow name cannot be empty".to_string());
    }

    if name.len() > 255 {
        return Err("Workflow name is too long (max 255 characters)".to_string());
    }

    // Prevent potential XSS in workflow names
    if name.contains('<') || name.contains('>') {
        return Err("Workflow name contains invalid characters".to_string());
    }

    Ok(())
}

pub fn validate_pagination(page: i64, limit: i64) -> Result<(i64, i64), String> {
    if page < 1 {
        return Err("Page must be greater than 0".to_string());
    }

    if limit < 1 || limit > 100 {
        return Err("Limit must be between 1 and 100".to_string());
    }

    Ok((page, limit))
}

pub fn validate_uuid(id: &str) -> Result<(), String> {
    uuid::Uuid::parse_str(id)
        .map(|_| ())
        .map_err(|_| "Invalid UUID format".to_string())
}

// Helper to create validation error responses
pub fn validation_error(message: &str) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": "Validation error",
            "message": message
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_url() {
        assert!(validate_url("https://example.com").is_ok());
        assert!(validate_url("http://api.example.com/v1/endpoint").is_ok());

        assert!(validate_url("").is_err());
        assert!(validate_url("not-a-url").is_err());
        assert!(validate_url("https://example.com/<script>").is_err());
    }

    #[test]
    fn test_validate_scan_type() {
        assert!(validate_scan_type("openapi").is_ok());
        assert!(validate_scan_type("auto").is_ok());
        assert!(validate_scan_type("rest").is_ok());

        assert!(validate_scan_type("invalid").is_err());
        assert!(validate_scan_type("").is_err());
    }

    #[test]
    fn test_validate_workflow_name() {
        assert!(validate_workflow_name("My Workflow").is_ok());
        assert!(validate_workflow_name("Test-123").is_ok());

        assert!(validate_workflow_name("").is_err());
        assert!(validate_workflow_name("<script>alert('xss')</script>").is_err());
    }

    #[test]
    fn test_validate_pagination() {
        assert!(validate_pagination(1, 20).is_ok());
        assert!(validate_pagination(5, 50).is_ok());

        assert!(validate_pagination(0, 20).is_err());
        assert!(validate_pagination(-1, 20).is_err());
        assert!(validate_pagination(1, 0).is_err());
        assert!(validate_pagination(1, 101).is_err());
    }

    #[test]
    fn test_validate_uuid() {
        assert!(validate_uuid("550e8400-e29b-41d4-a716-446655440000").is_ok());

        assert!(validate_uuid("not-a-uuid").is_err());
        assert!(validate_uuid("").is_err());
        assert!(validate_uuid("123").is_err());
    }
}
