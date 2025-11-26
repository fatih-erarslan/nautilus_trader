// use std::fmt; // Removed unused import

/// Common error type for the application
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Authorization error: {0}")]
    Authorization(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("External service error: {0}")]
    ExternalService(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Network error: {0}")]
    Network(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn status_code(&self) -> u16 {
        match self {
            Error::Database(_) => 500,
            Error::Authentication(_) => 401,
            Error::Authorization(_) => 403,
            Error::Validation(_) => 400,
            Error::NotFound(_) => 404,
            Error::ExternalService(_) => 502,
            Error::Internal(_) => 500,
            Error::Configuration(_) => 500,
            Error::Serialization(_) => 400,
            Error::Network(_) => 503,
        }
    }

    pub fn error_type(&self) -> &str {
        match self {
            Error::Database(_) => "database_error",
            Error::Authentication(_) => "authentication_error",
            Error::Authorization(_) => "authorization_error",
            Error::Validation(_) => "validation_error",
            Error::NotFound(_) => "not_found",
            Error::ExternalService(_) => "external_service_error",
            Error::Internal(_) => "internal_error",
            Error::Configuration(_) => "configuration_error",
            Error::Serialization(_) => "serialization_error",
            Error::Network(_) => "network_error",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_status_codes() {
        assert_eq!(Error::Database("test".into()).status_code(), 500);
        assert_eq!(Error::Authentication("test".into()).status_code(), 401);
        assert_eq!(Error::Authorization("test".into()).status_code(), 403);
        assert_eq!(Error::Validation("test".into()).status_code(), 400);
        assert_eq!(Error::NotFound("test".into()).status_code(), 404);
    }

    #[test]
    fn test_error_types() {
        assert_eq!(Error::Database("test".into()).error_type(), "database_error");
        assert_eq!(Error::Authentication("test".into()).error_type(), "authentication_error");
        assert_eq!(Error::Validation("test".into()).error_type(), "validation_error");
    }
}
