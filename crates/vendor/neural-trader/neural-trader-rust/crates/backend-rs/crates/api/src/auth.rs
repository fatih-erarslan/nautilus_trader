use axum::{
    async_trait,
    extract::{FromRequestParts, State},
    http::{request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json, RequestPartsExt,
};
use axum_extra::{
    headers::{authorization::Bearer, Authorization},
    TypedHeader,
};
use jsonwebtoken::{decode, decode_header, DecodingKey, Validation, Algorithm};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashSet;

/// User roles for RBAC
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    User,
    Guest,
    Scanner,
}

impl UserRole {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "admin" => Some(UserRole::Admin),
            "user" => Some(UserRole::User),
            "guest" => Some(UserRole::Guest),
            "scanner" => Some(UserRole::Scanner),
            _ => None,
        }
    }
}

/// JWT Claims structure matching Supabase token format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,           // User ID
    pub email: Option<String>, // User email
    pub role: String,          // User role from Supabase
    pub aud: Option<String>,   // Audience
    pub exp: usize,            // Expiration time
    pub iat: Option<usize>,    // Issued at
    pub iss: Option<String>,   // Issuer

    // Supabase specific fields
    pub app_metadata: Option<serde_json::Value>,
    pub user_metadata: Option<serde_json::Value>,
}

/// User context extracted from JWT token
#[derive(Debug, Clone)]
pub struct UserContext {
    pub user_id: String,
    pub email: Option<String>,
    pub roles: HashSet<UserRole>,
}

impl UserContext {
    /// Check if user has a specific role
    pub fn has_role(&self, role: &UserRole) -> bool {
        self.roles.contains(role)
    }

    /// Check if user has any of the specified roles
    pub fn has_any_role(&self, roles: &[UserRole]) -> bool {
        roles.iter().any(|role| self.roles.contains(role))
    }

    /// Check if user has all of the specified roles
    pub fn has_all_roles(&self, roles: &[UserRole]) -> bool {
        roles.iter().all(|role| self.roles.contains(role))
    }
}

/// Auth error types
#[derive(Debug)]
pub enum AuthError {
    MissingToken,
    InvalidToken,
    ExpiredToken,
    InsufficientPermissions,
    InvalidRole,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingToken => (StatusCode::UNAUTHORIZED, "Missing authentication token"),
            AuthError::InvalidToken => (StatusCode::UNAUTHORIZED, "Invalid authentication token"),
            AuthError::ExpiredToken => (StatusCode::UNAUTHORIZED, "Token has expired"),
            AuthError::InsufficientPermissions => {
                (StatusCode::FORBIDDEN, "Insufficient permissions")
            }
            AuthError::InvalidRole => (StatusCode::UNAUTHORIZED, "Invalid user role"),
        };

        (
            status,
            Json(json!({
                "error": message,
                "status": status.as_u16()
            })),
        )
            .into_response()
    }
}

/// Configuration for JWT validation
#[derive(Clone)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub jwt_algorithm: Algorithm,
    pub require_exp: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: std::env::var("SUPABASE_JWT_SECRET")
                .or_else(|_| std::env::var("JWT_SECRET"))
                .unwrap_or_else(|_| {
                    tracing::warn!("No JWT_SECRET set, using default (INSECURE for production!)");
                    "default-secret-change-in-production".to_string()
                }),
            jwt_algorithm: Algorithm::HS256,
            require_exp: true,
        }
    }
}

impl AuthConfig {
    pub fn from_env() -> Self {
        Self::default()
    }

    /// Validate and extract user context from JWT token
    pub fn validate_token(&self, token: &str) -> Result<UserContext, AuthError> {
        // First decode the header to check algorithm
        let header = decode_header(token).map_err(|e| {
            tracing::error!("Failed to decode token header: {}", e);
            AuthError::InvalidToken
        })?;

        // Verify algorithm matches
        if header.alg != self.jwt_algorithm {
            tracing::error!("Token algorithm mismatch: expected {:?}, got {:?}",
                self.jwt_algorithm, header.alg);
            return Err(AuthError::InvalidToken);
        }

        // Set up validation
        let mut validation = Validation::new(self.jwt_algorithm);
        validation.validate_exp = self.require_exp;
        validation.set_audience(&["authenticated"]); // Supabase default audience

        // Decode and validate token
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &validation,
        )
        .map_err(|e| {
            tracing::error!("Token validation failed: {}", e);
            match e.kind() {
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => AuthError::ExpiredToken,
                _ => AuthError::InvalidToken,
            }
        })?;

        let claims = token_data.claims;

        // Extract roles
        let mut roles = HashSet::new();

        // Parse primary role from claims
        if let Some(role) = UserRole::from_str(&claims.role) {
            roles.insert(role);
        } else {
            tracing::warn!("Unknown role '{}' for user {}", claims.role, claims.sub);
            // Default to Guest if role is unknown
            roles.insert(UserRole::Guest);
        }

        // Extract additional roles from app_metadata if present
        if let Some(app_metadata) = &claims.app_metadata {
            if let Some(extra_roles) = app_metadata.get("roles").and_then(|r| r.as_array()) {
                for role_value in extra_roles {
                    if let Some(role_str) = role_value.as_str() {
                        if let Some(role) = UserRole::from_str(role_str) {
                            roles.insert(role);
                        }
                    }
                }
            }
        }

        Ok(UserContext {
            user_id: claims.sub,
            email: claims.email,
            roles,
        })
    }
}

/// Extractor for authenticated user context
/// Usage: `async fn handler(user: AuthUser) { ... }`
#[async_trait]
impl<S> FromRequestParts<S> for UserContext
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Extract Authorization header
        let TypedHeader(Authorization(bearer)) = parts
            .extract::<TypedHeader<Authorization<Bearer>>>()
            .await
            .map_err(|_| {
                tracing::debug!("Missing or invalid Authorization header");
                AuthError::MissingToken
            })?;

        // Load auth config
        let config = AuthConfig::from_env();

        // Validate token and extract user context
        config.validate_token(bearer.token())
    }
}

/// Middleware for role-based access control
/// Usage: `Router::new().route(...).layer(axum::middleware::from_fn(require_role(UserRole::Admin)))`
pub async fn require_role(
    user: UserContext,
    required_role: UserRole,
) -> Result<UserContext, AuthError> {
    if user.has_role(&required_role) {
        Ok(user)
    } else {
        tracing::warn!(
            "User {} attempted to access resource requiring role {:?} but has roles {:?}",
            user.user_id,
            required_role,
            user.roles
        );
        Err(AuthError::InsufficientPermissions)
    }
}

/// Middleware for requiring any of multiple roles
pub async fn require_any_role(
    user: UserContext,
    required_roles: Vec<UserRole>,
) -> Result<UserContext, AuthError> {
    if user.has_any_role(&required_roles) {
        Ok(user)
    } else {
        tracing::warn!(
            "User {} attempted to access resource requiring any of {:?} but has roles {:?}",
            user.user_id,
            required_roles,
            user.roles
        );
        Err(AuthError::InsufficientPermissions)
    }
}

/// Helper function to create a test JWT token (for testing only!)
#[cfg(test)]
pub fn create_test_token(user_id: &str, role: &str, secret: &str) -> String {
    use jsonwebtoken::{encode, EncodingKey, Header};

    let claims = Claims {
        sub: user_id.to_string(),
        email: Some(format!("{}@example.com", user_id)),
        role: role.to_string(),
        aud: Some("authenticated".to_string()),
        exp: (chrono::Utc::now() + chrono::Duration::hours(1)).timestamp() as usize,
        iat: Some(chrono::Utc::now().timestamp() as usize),
        iss: Some("test-issuer".to_string()),
        app_metadata: None,
        user_metadata: None,
    };

    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_bytes()))
        .expect("Failed to create test token")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_role_from_str() {
        assert_eq!(UserRole::from_str("admin"), Some(UserRole::Admin));
        assert_eq!(UserRole::from_str("ADMIN"), Some(UserRole::Admin));
        assert_eq!(UserRole::from_str("user"), Some(UserRole::User));
        assert_eq!(UserRole::from_str("guest"), Some(UserRole::Guest));
        assert_eq!(UserRole::from_str("scanner"), Some(UserRole::Scanner));
        assert_eq!(UserRole::from_str("invalid"), None);
    }

    #[test]
    fn test_user_context_has_role() {
        let mut roles = HashSet::new();
        roles.insert(UserRole::Admin);
        roles.insert(UserRole::User);

        let context = UserContext {
            user_id: "test-user".to_string(),
            email: Some("test@example.com".to_string()),
            roles,
        };

        assert!(context.has_role(&UserRole::Admin));
        assert!(context.has_role(&UserRole::User));
        assert!(!context.has_role(&UserRole::Guest));
    }

    #[test]
    fn test_user_context_has_any_role() {
        let mut roles = HashSet::new();
        roles.insert(UserRole::User);

        let context = UserContext {
            user_id: "test-user".to_string(),
            email: Some("test@example.com".to_string()),
            roles,
        };

        assert!(context.has_any_role(&[UserRole::Admin, UserRole::User]));
        assert!(!context.has_any_role(&[UserRole::Admin, UserRole::Guest]));
    }

    #[test]
    fn test_validate_token_success() {
        let secret = "test-secret";
        let config = AuthConfig {
            jwt_secret: secret.to_string(),
            jwt_algorithm: Algorithm::HS256,
            require_exp: true,
        };

        let token = create_test_token("user123", "admin", secret);
        let result = config.validate_token(&token);

        assert!(result.is_ok());
        let context = result.unwrap();
        assert_eq!(context.user_id, "user123");
        assert!(context.has_role(&UserRole::Admin));
    }

    #[test]
    fn test_validate_token_invalid_secret() {
        let config = AuthConfig {
            jwt_secret: "wrong-secret".to_string(),
            jwt_algorithm: Algorithm::HS256,
            require_exp: true,
        };

        let token = create_test_token("user123", "admin", "correct-secret");
        let result = config.validate_token(&token);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AuthError::InvalidToken));
    }

    #[test]
    fn test_validate_token_expired() {
        use jsonwebtoken::{encode, EncodingKey, Header};

        let secret = "test-secret";
        let config = AuthConfig {
            jwt_secret: secret.to_string(),
            jwt_algorithm: Algorithm::HS256,
            require_exp: true,
        };

        // Create expired token
        let claims = Claims {
            sub: "user123".to_string(),
            email: Some("user@example.com".to_string()),
            role: "user".to_string(),
            aud: Some("authenticated".to_string()),
            exp: (chrono::Utc::now() - chrono::Duration::hours(1)).timestamp() as usize,
            iat: Some(chrono::Utc::now().timestamp() as usize),
            iss: Some("test-issuer".to_string()),
            app_metadata: None,
            user_metadata: None,
        };

        let token = encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_bytes()))
            .unwrap();

        let result = config.validate_token(&token);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AuthError::ExpiredToken));
    }
}
