use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,      // Subject (user ID)
    pub exp: usize,       // Expiration time
    pub iat: usize,       // Issued at
    pub role: String,     // User role
}

#[derive(Clone)]
pub struct JwtConfig {
    pub secret: String,
    pub expiration_hours: i64,
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: std::env::var("JWT_SECRET")
                .unwrap_or_else(|_| "default-secret-change-in-production".to_string()),
            expiration_hours: 24,
        }
    }
}

impl JwtConfig {
    pub fn new(secret: String, expiration_hours: i64) -> Self {
        Self {
            secret,
            expiration_hours,
        }
    }

    pub fn create_token(&self, user_id: &str, role: &str) -> Result<String, jsonwebtoken::errors::Error> {
        let now = chrono::Utc::now();
        let exp = (now + chrono::Duration::hours(self.expiration_hours)).timestamp() as usize;
        let iat = now.timestamp() as usize;

        let claims = Claims {
            sub: user_id.to_string(),
            exp,
            iat,
            role: role.to_string(),
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.secret.as_bytes()),
        )
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
        let validation = Validation::default();
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.secret.as_bytes()),
            &validation,
        )?;
        Ok(token_data.claims)
    }
}

// Middleware to require authentication on protected routes
pub async fn auth_middleware(
    State(jwt_config): State<Arc<JwtConfig>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let auth_header = request
        .headers()
        .get("Authorization")
        .and_then(|header| header.to_str().ok());

    let token = if let Some(auth_header) = auth_header {
        if auth_header.starts_with("Bearer ") {
            &auth_header[7..]
        } else {
            return Err(StatusCode::UNAUTHORIZED);
        }
    } else {
        return Err(StatusCode::UNAUTHORIZED);
    };

    match jwt_config.validate_token(token) {
        Ok(claims) => {
            // Add claims to request extensions for use in handlers
            request.extensions_mut().insert(claims);
            Ok(next.run(request).await)
        }
        Err(_) => Err(StatusCode::UNAUTHORIZED),
    }
}

// Middleware to check specific roles
pub async fn require_role(
    required_role: &'static str,
    claims: Option<&Claims>,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    match claims {
        Some(claims) if claims.role == required_role || claims.role == "admin" => Ok(()),
        _ => Err((
            StatusCode::FORBIDDEN,
            Json(json!({
                "error": "Insufficient permissions",
                "required_role": required_role
            })),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_validate_token() {
        let config = JwtConfig::new("test-secret".to_string(), 24);
        let token = config.create_token("user123", "user").unwrap();
        let claims = config.validate_token(&token).unwrap();

        assert_eq!(claims.sub, "user123");
        assert_eq!(claims.role, "user");
    }

    #[test]
    fn test_invalid_token() {
        let config = JwtConfig::new("test-secret".to_string(), 24);
        let result = config.validate_token("invalid.token.here");

        assert!(result.is_err());
    }

    #[test]
    fn test_token_with_wrong_secret() {
        let config1 = JwtConfig::new("secret1".to_string(), 24);
        let config2 = JwtConfig::new("secret2".to_string(), 24);

        let token = config1.create_token("user123", "user").unwrap();
        let result = config2.validate_token(&token);

        assert!(result.is_err());
    }
}
