//! Authentication and Authorization Module
//!
//! Provides comprehensive security features:
//! - API key validation
//! - JWT token generation and validation
//! - Role-Based Access Control (RBAC)
//! - User authentication
//! - Session management

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

/// User roles for RBAC
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[napi]
pub enum UserRole {
    /// Read-only access
    ReadOnly,
    /// Standard user with trading capabilities
    User,
    /// Administrative access
    Admin,
    /// System service account
    Service,
}

impl UserRole {
    /// Check if role has permission for an action
    pub fn has_permission(&self, required_role: UserRole) -> bool {
        match (self, required_role) {
            (UserRole::Admin, _) => true,
            (UserRole::User, UserRole::ReadOnly) => true,
            (UserRole::User, UserRole::User) => true,
            (UserRole::Service, UserRole::Service) => true,
            (UserRole::ReadOnly, UserRole::ReadOnly) => true,
            _ => false,
        }
    }

    /// Get permission level as numeric value (higher = more permissions)
    pub fn permission_level(&self) -> u8 {
        match self {
            UserRole::ReadOnly => 1,
            UserRole::User => 2,
            UserRole::Service => 3,
            UserRole::Admin => 4,
        }
    }
}

/// Authenticated user information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct AuthUser {
    pub user_id: String,
    pub username: String,
    pub role: UserRole,
    pub api_key: String,
    pub created_at: String,
    pub last_activity: String,
}

/// API Key with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key: String,
    pub user_id: String,
    pub username: String,
    pub role: UserRole,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub rate_limit: u32, // requests per minute
    pub allowed_operations: Vec<String>,
}

impl ApiKey {
    /// Check if API key is valid and not expired
    pub fn is_valid(&self) -> bool {
        if !self.is_active {
            return false;
        }

        if let Some(expires) = self.expires_at {
            if Utc::now() > expires {
                return false;
            }
        }

        true
    }

    /// Check if operation is allowed for this API key
    pub fn can_perform(&self, operation: &str) -> bool {
        if !self.is_valid() {
            return false;
        }

        // Admin can do everything
        if self.role == UserRole::Admin {
            return true;
        }

        // Check allowed operations list
        self.allowed_operations.is_empty() ||
        self.allowed_operations.contains(&operation.to_string()) ||
        self.allowed_operations.contains(&"*".to_string())
    }
}

/// JWT Claims structure
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,      // Subject (user_id)
    pub username: String,
    pub role: UserRole,
    pub exp: i64,         // Expiration time
    pub iat: i64,         // Issued at
    pub jti: String,      // JWT ID (unique identifier)
}

/// Authentication manager - central security component
pub struct AuthManager {
    api_keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    jwt_secret: String,
    token_expiry_hours: i64,
}

impl AuthManager {
    /// Create new authentication manager
    pub fn new(jwt_secret: String) -> Self {
        Self {
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            jwt_secret,
            token_expiry_hours: 24,
        }
    }

    /// Generate a new API key for a user
    pub fn generate_api_key(
        &self,
        user_id: String,
        username: String,
        role: UserRole,
        rate_limit: u32,
        expires_in_days: Option<i64>,
    ) -> Result<String> {
        let api_key = format!("ntk_{}", Uuid::new_v4().to_string().replace("-", ""));

        let key_data = ApiKey {
            key: api_key.clone(),
            user_id,
            username,
            role,
            created_at: Utc::now(),
            expires_at: expires_in_days.map(|days| Utc::now() + Duration::days(days)),
            is_active: true,
            rate_limit,
            allowed_operations: vec!["*".to_string()], // All operations by default
        };

        let mut keys = self.api_keys.write()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        keys.insert(api_key.clone(), key_data);

        Ok(api_key)
    }

    /// Validate an API key and return associated user info
    pub fn validate_api_key(&self, api_key: &str) -> Result<ApiKey> {
        let keys = self.api_keys.read()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        let key_data = keys.get(api_key)
            .ok_or_else(|| Error::from_reason("Invalid API key"))?;

        if !key_data.is_valid() {
            return Err(Error::from_reason("API key expired or inactive"));
        }

        Ok(key_data.clone())
    }

    /// Revoke an API key
    pub fn revoke_api_key(&self, api_key: &str) -> Result<()> {
        let mut keys = self.api_keys.write()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        if let Some(key_data) = keys.get_mut(api_key) {
            key_data.is_active = false;
            Ok(())
        } else {
            Err(Error::from_reason("API key not found"))
        }
    }

    /// Generate JWT token for authenticated user
    pub fn generate_jwt_token(&self, user_id: &str, username: &str, role: UserRole) -> Result<String> {
        use jsonwebtoken::{encode, Header, EncodingKey};

        let now = Utc::now();
        let exp = now + Duration::hours(self.token_expiry_hours);

        let claims = Claims {
            sub: user_id.to_string(),
            username: username.to_string(),
            role,
            exp: exp.timestamp(),
            iat: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(self.jwt_secret.as_bytes()),
        )
        .map_err(|e| Error::from_reason(format!("JWT generation failed: {}", e)))
    }

    /// Validate and decode JWT token
    pub fn validate_jwt_token(&self, token: &str) -> Result<Claims> {
        use jsonwebtoken::{decode, DecodingKey, Validation};

        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(self.jwt_secret.as_bytes()),
            &Validation::default(),
        )
        .map_err(|e| Error::from_reason(format!("Invalid JWT token: {}", e)))?;

        Ok(token_data.claims)
    }

    /// Check if user has required permission for operation
    pub fn authorize(&self, api_key: &str, required_role: UserRole, operation: &str) -> Result<ApiKey> {
        let key_data = self.validate_api_key(api_key)?;

        // Check role permission
        if !key_data.role.has_permission(required_role) {
            return Err(Error::from_reason(format!(
                "Insufficient permissions: requires {:?}, has {:?}",
                required_role, key_data.role
            )));
        }

        // Check operation permission
        if !key_data.can_perform(operation) {
            return Err(Error::from_reason(format!(
                "Operation '{}' not allowed for this API key",
                operation
            )));
        }

        Ok(key_data)
    }

    /// List all API keys (admin only)
    pub fn list_api_keys(&self, admin_key: &str) -> Result<Vec<ApiKey>> {
        let admin = self.validate_api_key(admin_key)?;

        if admin.role != UserRole::Admin {
            return Err(Error::from_reason("Admin access required"));
        }

        let keys = self.api_keys.read()
            .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

        Ok(keys.values().cloned().collect())
    }

    /// Get API key usage statistics
    pub fn get_key_stats(&self, api_key: &str) -> Result<ApiKeyStats> {
        let key_data = self.validate_api_key(api_key)?;

        Ok(ApiKeyStats {
            key_id: key_data.key[..12].to_string() + "...", // Masked
            username: key_data.username,
            role: key_data.role,
            created_at: key_data.created_at.to_rfc3339(),
            expires_at: key_data.expires_at.map(|d| d.to_rfc3339()),
            is_active: key_data.is_active,
            rate_limit: key_data.rate_limit,
        })
    }
}

/// API Key statistics
#[derive(Debug, Serialize, Deserialize)]
#[napi(object)]
pub struct ApiKeyStats {
    pub key_id: String,
    pub username: String,
    pub role: UserRole,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub is_active: bool,
    pub rate_limit: u32,
}

/// Global authentication manager instance
static AUTH_MANAGER: once_cell::sync::Lazy<Arc<RwLock<Option<AuthManager>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize authentication system
#[napi]
pub fn init_auth(jwt_secret: Option<String>) -> Result<String> {
    let secret = jwt_secret.unwrap_or_else(|| {
        std::env::var("JWT_SECRET").expect(
            "SECURITY ERROR: JWT_SECRET environment variable must be set.\n\
             Generate a secure secret with: openssl rand -hex 64\n\
             Set it in your environment or .env file before starting the application."
        )
    });

    let manager = AuthManager::new(secret);

    let mut auth = AUTH_MANAGER.write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    *auth = Some(manager);

    Ok("Authentication system initialized".to_string())
}

/// Create a new API key
#[napi]
pub fn create_api_key(
    username: String,
    role: String,
    rate_limit: Option<u32>,
    expires_in_days: Option<i64>,
) -> Result<String> {
    let auth = AUTH_MANAGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let manager = auth.as_ref()
        .ok_or_else(|| Error::from_reason("Auth system not initialized"))?;

    let user_role = match role.as_str() {
        "readonly" => UserRole::ReadOnly,
        "user" => UserRole::User,
        "admin" => UserRole::Admin,
        "service" => UserRole::Service,
        _ => return Err(Error::from_reason("Invalid role")),
    };

    let user_id = Uuid::new_v4().to_string();

    manager.generate_api_key(
        user_id,
        username,
        user_role,
        rate_limit.unwrap_or(100), // Default 100 req/min
        expires_in_days,
    )
}

/// Validate an API key
#[napi]
pub fn validate_api_key(api_key: String) -> Result<AuthUser> {
    let auth = AUTH_MANAGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let manager = auth.as_ref()
        .ok_or_else(|| Error::from_reason("Auth system not initialized"))?;

    let key_data = manager.validate_api_key(&api_key)?;

    Ok(AuthUser {
        user_id: key_data.user_id,
        username: key_data.username,
        role: key_data.role,
        api_key: key_data.key[..12].to_string() + "...", // Masked
        created_at: key_data.created_at.to_rfc3339(),
        last_activity: Utc::now().to_rfc3339(),
    })
}

/// Revoke an API key
#[napi]
pub fn revoke_api_key(api_key: String) -> Result<String> {
    let auth = AUTH_MANAGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let manager = auth.as_ref()
        .ok_or_else(|| Error::from_reason("Auth system not initialized"))?;

    manager.revoke_api_key(&api_key)?;

    Ok("API key revoked successfully".to_string())
}

/// Generate JWT token
#[napi]
pub fn generate_token(api_key: String) -> Result<String> {
    let auth = AUTH_MANAGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let manager = auth.as_ref()
        .ok_or_else(|| Error::from_reason("Auth system not initialized"))?;

    let key_data = manager.validate_api_key(&api_key)?;

    manager.generate_jwt_token(&key_data.user_id, &key_data.username, key_data.role)
}

/// Validate JWT token
#[napi]
pub fn validate_token(token: String) -> Result<AuthUser> {
    let auth = AUTH_MANAGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let manager = auth.as_ref()
        .ok_or_else(|| Error::from_reason("Auth system not initialized"))?;

    let claims = manager.validate_jwt_token(&token)?;

    Ok(AuthUser {
        user_id: claims.sub,
        username: claims.username,
        role: claims.role,
        api_key: "***".to_string(),
        created_at: DateTime::from_timestamp(claims.iat, 0)
            .ok_or_else(|| Error::from_reason("Invalid timestamp"))?
            .to_rfc3339(),
        last_activity: Utc::now().to_rfc3339(),
    })
}

/// Check authorization for an operation
#[napi]
pub fn check_authorization(api_key: String, operation: String, required_role: String) -> Result<bool> {
    let auth = AUTH_MANAGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let manager = auth.as_ref()
        .ok_or_else(|| Error::from_reason("Auth system not initialized"))?;

    let role = match required_role.as_str() {
        "readonly" => UserRole::ReadOnly,
        "user" => UserRole::User,
        "admin" => UserRole::Admin,
        "service" => UserRole::Service,
        _ => return Err(Error::from_reason("Invalid role")),
    };

    match manager.authorize(&api_key, role, &operation) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_role_permissions() {
        assert!(UserRole::Admin.has_permission(UserRole::User));
        assert!(UserRole::Admin.has_permission(UserRole::ReadOnly));
        assert!(UserRole::User.has_permission(UserRole::ReadOnly));
        assert!(!UserRole::ReadOnly.has_permission(UserRole::User));
        assert!(!UserRole::User.has_permission(UserRole::Admin));
    }

    #[test]
    fn test_api_key_generation() {
        let manager = AuthManager::new("test_secret".to_string());
        let key = manager.generate_api_key(
            "user123".to_string(),
            "testuser".to_string(),
            UserRole::User,
            100,
            Some(30),
        ).unwrap();

        assert!(key.starts_with("ntk_"));
        assert!(manager.validate_api_key(&key).is_ok());
    }

    #[test]
    fn test_api_key_revocation() {
        let manager = AuthManager::new("test_secret".to_string());
        let key = manager.generate_api_key(
            "user123".to_string(),
            "testuser".to_string(),
            UserRole::User,
            100,
            Some(30),
        ).unwrap();

        assert!(manager.validate_api_key(&key).is_ok());
        assert!(manager.revoke_api_key(&key).is_ok());
        assert!(manager.validate_api_key(&key).is_err());
    }
}
