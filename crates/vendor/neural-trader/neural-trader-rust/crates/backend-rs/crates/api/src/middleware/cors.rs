use tower_http::cors::{Any, CorsLayer};
use std::time::Duration;

pub fn configure_cors() -> CorsLayer {
    // Get allowed origins from environment variable or use defaults
    let allowed_origins = std::env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:3000,http://localhost:5173".to_string());

    let origins: Vec<_> = allowed_origins
        .split(',')
        .filter_map(|origin| origin.trim().parse::<axum::http::HeaderValue>().ok())
        .collect();

    // If no valid origins configured, use permissive mode for development
    // WARNING: This should never be used in production!
    if origins.is_empty() {
        tracing::warn!("⚠️  Using permissive CORS - NOT SUITABLE FOR PRODUCTION");
        return CorsLayer::permissive();
    }

    tracing::info!("✅ CORS configured with specific allowed origins");

    CorsLayer::new()
        .allow_origin(origins)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::PUT,
            axum::http::Method::DELETE,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
            axum::http::header::ACCEPT,
        ])
        .allow_credentials(true)
        .max_age(Duration::from_secs(3600))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cors_configuration() {
        // Test that we can create a CORS layer
        let cors = configure_cors();
        // Basic test - just ensure it doesn't panic
        assert!(true);
    }
}
