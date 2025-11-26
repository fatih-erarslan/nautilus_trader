/**
 * JWT Security Module - Production Hardened
 *
 * Fixes CRITICAL-1 vulnerability: Hardcoded JWT secret
 * Risk: 10/10 - CATASTROPHIC
 *
 * Changes:
 * 1. Remove default JWT secret
 * 2. Enforce environment variable requirement
 * 3. Add secret strength validation
 * 4. Panic if insecure configuration detected
 */

use std::env;
use jsonwebtoken::{Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JwtError {
    #[error("JWT_SECRET environment variable not set - refusing to start")]
    SecretNotSet,

    #[error("JWT secret is too weak (minimum 32 characters required, got {0})")]
    SecretTooWeak(usize),

    #[error("JWT secret uses insecure default value - refusing to start")]
    InsecureDefault,

    #[error("JWT encoding error: {0}")]
    EncodingError(#[from] jsonwebtoken::errors::Error),

    #[error("JWT validation error: {0}")]
    ValidationError(String),
}

/// Secure JWT configuration with mandatory secret validation
#[derive(Clone)]
pub struct SecureJwtConfig {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    validation: Validation,
    expiration_hours: i64,
}

impl SecureJwtConfig {
    /// Create new JWT config with strict security requirements
    ///
    /// # Security Requirements
    /// 1. JWT_SECRET must be set as environment variable
    /// 2. Secret must be at least 32 characters long
    /// 3. Secret must not match any known default values
    /// 4. Will panic if insecure configuration detected
    ///
    /// # Panics
    /// This function will panic if:
    /// - JWT_SECRET environment variable is not set
    /// - Secret is less than 32 characters
    /// - Secret matches known insecure defaults
    pub fn from_env() -> Result<Self, JwtError> {
        // 1. Check if JWT_SECRET is set
        let secret = env::var("JWT_SECRET").map_err(|_| {
            eprintln!("❌ FATAL SECURITY ERROR: JWT_SECRET environment variable not set");
            eprintln!("   Set a strong secret in production:");
            eprintln!("   export JWT_SECRET=$(openssl rand -base64 64)");
            JwtError::SecretNotSet
        })?;

        // 2. Validate secret strength (minimum 32 characters)
        if secret.len() < 32 {
            eprintln!("❌ FATAL SECURITY ERROR: JWT secret too weak");
            eprintln!("   Current length: {} characters", secret.len());
            eprintln!("   Minimum required: 32 characters");
            eprintln!("   Generate a strong secret:");
            eprintln!("   export JWT_SECRET=$(openssl rand -base64 64)");
            return Err(JwtError::SecretTooWeak(secret.len()));
        }

        // 3. Check for known insecure defaults
        let insecure_defaults = [
            "default-secret-change-in-production",
            "your-secret-key-change-in-production",
            "change-me-in-production",
            "secret",
            "supersecret",
            "password",
            "12345",
        ];

        for insecure in &insecure_defaults {
            if secret.contains(insecure) {
                eprintln!("❌ FATAL SECURITY ERROR: JWT secret uses insecure default value");
                eprintln!("   Detected pattern: {}", insecure);
                eprintln!("   Generate a cryptographically secure secret:");
                eprintln!("   export JWT_SECRET=$(openssl rand -base64 64)");
                return Err(JwtError::InsecureDefault);
            }
        }

        // 4. Log success (without exposing secret)
        eprintln!("✅ JWT configuration validated (secret length: {} chars)", secret.len());

        let encoding_key = EncodingKey::from_secret(secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(secret.as_bytes());

        let mut validation = Validation::new(Algorithm::HS256);
        validation.validate_exp = true;
        validation.leeway = 60; // 60 seconds clock skew tolerance

        Ok(Self {
            encoding_key,
            decoding_key,
            validation,
            expiration_hours: 24, // Default 24 hours
        })
    }

    /// Create JWT config with custom expiration
    pub fn with_expiration(mut self, hours: i64) -> Self {
        self.expiration_hours = hours;
        self
    }

    /// Encode JWT token with claims
    pub fn encode<T: Serialize>(&self, claims: &T) -> Result<String, JwtError> {
        let header = Header::new(Algorithm::HS256);
        let token = jsonwebtoken::encode(&header, claims, &self.encoding_key)?;
        Ok(token)
    }

    /// Decode and validate JWT token
    pub fn decode<T: for<'de> Deserialize<'de>>(&self, token: &str) -> Result<T, JwtError> {
        let token_data = jsonwebtoken::decode::<T>(token, &self.decoding_key, &self.validation)
            .map_err(|e| JwtError::ValidationError(e.to_string()))?;
        Ok(token_data.claims)
    }

    /// Get expiration timestamp (current time + expiration_hours)
    pub fn get_expiration(&self) -> i64 {
        let now = chrono::Utc::now();
        let expiration = now + chrono::Duration::hours(self.expiration_hours);
        expiration.timestamp()
    }
}

/// Standard JWT claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,

    /// Issued at (Unix timestamp)
    pub iat: i64,

    /// Expiration (Unix timestamp)
    pub exp: i64,

    /// Custom claims
    #[serde(flatten)]
    pub custom: serde_json::Value,
}

impl Claims {
    /// Create new claims with user ID and expiration
    pub fn new(user_id: String, expiration_hours: i64) -> Self {
        let now = chrono::Utc::now();
        let iat = now.timestamp();
        let exp = (now + chrono::Duration::hours(expiration_hours)).timestamp();

        Self {
            sub: user_id,
            iat,
            exp,
            custom: serde_json::json!({}),
        }
    }

    /// Add custom claim
    pub fn with_claim(mut self, key: &str, value: serde_json::Value) -> Self {
        if let serde_json::Value::Object(ref mut map) = self.custom {
            map.insert(key.to_string(), value);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "JWT_SECRET environment variable not set")]
    fn test_missing_secret_panics() {
        // Clear environment variable
        env::remove_var("JWT_SECRET");

        // Should panic
        let _ = SecureJwtConfig::from_env().unwrap();
    }

    #[test]
    #[should_panic(expected = "JWT secret too weak")]
    fn test_weak_secret_panics() {
        // Set weak secret
        env::set_var("JWT_SECRET", "short");

        // Should panic
        let _ = SecureJwtConfig::from_env().unwrap();
    }

    #[test]
    #[should_panic(expected = "insecure default value")]
    fn test_insecure_default_panics() {
        // Set insecure default
        env::set_var("JWT_SECRET", "default-secret-change-in-production");

        // Should panic
        let _ = SecureJwtConfig::from_env().unwrap();
    }

    #[test]
    fn test_secure_secret_works() {
        // Set strong secret (64 character random string)
        env::set_var(
            "JWT_SECRET",
            "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        );

        // Should succeed
        let config = SecureJwtConfig::from_env().unwrap();

        // Test encoding/decoding
        let claims = Claims::new("user123".to_string(), 24);
        let token = config.encode(&claims).unwrap();
        let decoded: Claims = config.decode(&token).unwrap();

        assert_eq!(decoded.sub, "user123");
    }
}
