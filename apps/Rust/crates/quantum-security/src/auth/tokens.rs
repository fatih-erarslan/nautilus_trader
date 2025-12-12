//! Token-based Authentication
//!
//! This module provides token-based authentication implementations.

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Token Authentication Provider
#[derive(Debug, Clone)]
pub struct TokenProvider {
    pub id: Uuid,
    pub name: String,
    pub tokens: HashMap<String, AuthToken>,
    pub refresh_tokens: HashMap<String, RefreshToken>,
    pub token_config: TokenConfig,
    pub enabled: bool,
}

/// Authentication Token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub token_id: String,
    pub agent_id: String,
    pub token_type: TokenType,
    pub token_data: Vec<u8>,
    pub claims: HashMap<String, String>,
    pub issued_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub scopes: Vec<String>,
    pub revoked: bool,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}

/// Refresh Token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshToken {
    pub token_id: String,
    pub agent_id: String,
    pub token_data: Vec<u8>,
    pub issued_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub revoked: bool,
    pub uses_remaining: Option<u32>,
}

/// Token Type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TokenType {
    AccessToken,
    RefreshToken,
    SessionToken,
    APIKey,
    BearerToken,
    JWTToken,
    QuantumToken,
}

/// Token Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    pub access_token_lifetime: chrono::Duration,
    pub refresh_token_lifetime: chrono::Duration,
    pub session_token_lifetime: chrono::Duration,
    pub max_refresh_uses: Option<u32>,
    pub require_secure_transport: bool,
    pub enable_token_rotation: bool,
    pub quantum_signature_required: bool,
}

/// Token Validation Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenValidationResult {
    pub valid: bool,
    pub token_id: String,
    pub agent_id: String,
    pub scopes: Vec<String>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub error_message: Option<String>,
    pub warnings: Vec<String>,
}

/// Token Issuance Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenIssuanceRequest {
    pub agent_id: String,
    pub token_type: TokenType,
    pub scopes: Vec<String>,
    pub claims: HashMap<String, String>,
    pub custom_lifetime: Option<chrono::Duration>,
}

impl TokenProvider {
    /// Create new token provider
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            tokens: HashMap::new(),
            refresh_tokens: HashMap::new(),
            token_config: TokenConfig::default(),
            enabled: true,
        }
    }

    /// Issue new token
    pub async fn issue_token(
        &mut self,
        request: TokenIssuanceRequest,
    ) -> Result<AuthToken, QuantumSecurityError> {
        let now = chrono::Utc::now();
        let token_id = format!("token_{}", Uuid::new_v4());

        // Determine token lifetime
        let lifetime = request.custom_lifetime.unwrap_or_else(|| match request.token_type {
            TokenType::AccessToken => self.token_config.access_token_lifetime,
            TokenType::RefreshToken => self.token_config.refresh_token_lifetime,
            TokenType::SessionToken => self.token_config.session_token_lifetime,
            _ => self.token_config.access_token_lifetime,
        });

        // Generate token data (in real implementation, this would be cryptographically secure)
        let token_data = self.generate_token_data(&request.agent_id, &request.token_type)?;

        let token = AuthToken {
            token_id: token_id.clone(),
            agent_id: request.agent_id,
            token_type: request.token_type,
            token_data,
            claims: request.claims,
            issued_at: now,
            expires_at: now + lifetime,
            scopes: request.scopes,
            revoked: false,
            last_used: None,
        };

        self.tokens.insert(token_id, token.clone());

        Ok(token)
    }

    /// Validate token
    pub async fn validate_token(&mut self, token_data: &[u8]) -> Result<TokenValidationResult, QuantumSecurityError> {
        // Find token by data
        let token = self.tokens.values_mut()
            .find(|t| t.token_data == token_data)
            .ok_or_else(|| QuantumSecurityError::TokenNotFound("Token not found".to_string()))?;

        let mut warnings = Vec::new();

        // Check if revoked
        if token.revoked {
            return Ok(TokenValidationResult {
                valid: false,
                token_id: token.token_id.clone(),
                agent_id: token.agent_id.clone(),
                scopes: vec![],
                expires_at: token.expires_at,
                error_message: Some("Token is revoked".to_string()),
                warnings,
            });
        }

        // Check expiration
        let now = chrono::Utc::now();
        if now > token.expires_at {
            return Ok(TokenValidationResult {
                valid: false,
                token_id: token.token_id.clone(),
                agent_id: token.agent_id.clone(),
                scopes: vec![],
                expires_at: token.expires_at,
                error_message: Some("Token is expired".to_string()),
                warnings,
            });
        }

        // Check if expiring soon
        if now + chrono::Duration::minutes(5) > token.expires_at {
            warnings.push("Token expires within 5 minutes".to_string());
        }

        // Update last used
        token.last_used = Some(now);

        Ok(TokenValidationResult {
            valid: true,
            token_id: token.token_id.clone(),
            agent_id: token.agent_id.clone(),
            scopes: token.scopes.clone(),
            expires_at: token.expires_at,
            error_message: None,
            warnings,
        })
    }

    /// Refresh token
    pub async fn refresh_token(&mut self, refresh_token_data: &[u8]) -> Result<AuthToken, QuantumSecurityError> {
        // Find refresh token
        let refresh_token = self.refresh_tokens.values_mut()
            .find(|t| t.token_data == refresh_token_data)
            .ok_or_else(|| QuantumSecurityError::TokenNotFound("Refresh token not found".to_string()))?;

        // Check if revoked
        if refresh_token.revoked {
            return Err(QuantumSecurityError::TokenRevoked("Refresh token is revoked".to_string()));
        }

        // Check expiration
        let now = chrono::Utc::now();
        if now > refresh_token.expires_at {
            return Err(QuantumSecurityError::TokenExpired("Refresh token is expired".to_string()));
        }

        // Check usage limit
        if let Some(uses_remaining) = refresh_token.uses_remaining.as_mut() {
            if *uses_remaining == 0 {
                return Err(QuantumSecurityError::TokenExhausted("Refresh token is exhausted".to_string()));
            }
            *uses_remaining -= 1;
        }

        // Issue new access token
        let request = TokenIssuanceRequest {
            agent_id: refresh_token.agent_id.clone(),
            token_type: TokenType::AccessToken,
            scopes: vec!["default".to_string()],
            claims: HashMap::new(),
            custom_lifetime: None,
        };

        let new_token = self.issue_token(request).await?;

        // Optionally rotate refresh token
        if self.token_config.enable_token_rotation {
            refresh_token.revoked = true;
            self.issue_refresh_token(&refresh_token.agent_id).await?;
        }

        Ok(new_token)
    }

    /// Issue refresh token
    pub async fn issue_refresh_token(&mut self, agent_id: &str) -> Result<RefreshToken, QuantumSecurityError> {
        let now = chrono::Utc::now();
        let token_id = format!("refresh_{}", Uuid::new_v4());

        let refresh_token = RefreshToken {
            token_id: token_id.clone(),
            agent_id: agent_id.to_string(),
            token_data: self.generate_token_data(agent_id, &TokenType::RefreshToken)?,
            issued_at: now,
            expires_at: now + self.token_config.refresh_token_lifetime,
            revoked: false,
            uses_remaining: self.token_config.max_refresh_uses,
        };

        self.refresh_tokens.insert(token_id, refresh_token.clone());

        Ok(refresh_token)
    }

    /// Revoke token
    pub async fn revoke_token(&mut self, token_id: &str) -> Result<(), QuantumSecurityError> {
        let token = self.tokens.get_mut(token_id)
            .ok_or_else(|| QuantumSecurityError::TokenNotFound("Token not found".to_string()))?;

        token.revoked = true;
        Ok(())
    }

    /// Revoke all tokens for agent
    pub async fn revoke_all_tokens(&mut self, agent_id: &str) -> Result<(), QuantumSecurityError> {
        // Revoke access tokens
        for token in self.tokens.values_mut() {
            if token.agent_id == agent_id {
                token.revoked = true;
            }
        }

        // Revoke refresh tokens
        for token in self.refresh_tokens.values_mut() {
            if token.agent_id == agent_id {
                token.revoked = true;
            }
        }

        Ok(())
    }

    /// Get token info
    pub async fn get_token_info(&self, token_id: &str) -> Result<AuthToken, QuantumSecurityError> {
        self.tokens.get(token_id)
            .cloned()
            .ok_or_else(|| QuantumSecurityError::TokenNotFound("Token not found".to_string()))
    }

    /// List tokens for agent
    pub async fn list_tokens(&self, agent_id: &str) -> Result<Vec<AuthToken>, QuantumSecurityError> {
        let tokens = self.tokens.values()
            .filter(|t| t.agent_id == agent_id && !t.revoked)
            .cloned()
            .collect();
        Ok(tokens)
    }

    /// Clean up expired tokens
    pub async fn cleanup_expired_tokens(&mut self) -> Result<(), QuantumSecurityError> {
        let now = chrono::Utc::now();

        // Remove expired access tokens
        self.tokens.retain(|_, token| now <= token.expires_at);

        // Remove expired refresh tokens
        self.refresh_tokens.retain(|_, token| now <= token.expires_at);

        Ok(())
    }

    /// Generate token data
    fn generate_token_data(&self, agent_id: &str, token_type: &TokenType) -> Result<Vec<u8>, QuantumSecurityError> {
        // In real implementation, this would use cryptographically secure random generation
        // and proper signing/encryption
        let mut data = Vec::new();
        data.extend(agent_id.as_bytes());
        data.extend(format!("{:?}", token_type).as_bytes());
        data.extend(chrono::Utc::now().timestamp().to_be_bytes());
        data.extend(Uuid::new_v4().as_bytes());

        Ok(data)
    }
}

impl TokenConfig {
    /// Create default token configuration
    pub fn default() -> Self {
        Self {
            access_token_lifetime: chrono::Duration::minutes(15),
            refresh_token_lifetime: chrono::Duration::days(30),
            session_token_lifetime: chrono::Duration::hours(8),
            max_refresh_uses: Some(100),
            require_secure_transport: true,
            enable_token_rotation: true,
            quantum_signature_required: false,
        }
    }

    /// Create high-security configuration
    pub fn high_security() -> Self {
        Self {
            access_token_lifetime: chrono::Duration::minutes(5),
            refresh_token_lifetime: chrono::Duration::days(1),
            session_token_lifetime: chrono::Duration::hours(2),
            max_refresh_uses: Some(10),
            require_secure_transport: true,
            enable_token_rotation: true,
            quantum_signature_required: true,
        }
    }
}

impl Default for TokenProvider {
    fn default() -> Self {
        Self::new("default-token-provider".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_token_provider_creation() {
        let provider = TokenProvider::new("test".to_string());
        assert_eq!(provider.name, "test");
        assert!(provider.enabled);
    }

    #[tokio::test]
    async fn test_token_issuance() {
        let mut provider = TokenProvider::new("test".to_string());
        
        let request = TokenIssuanceRequest {
            agent_id: "test_agent".to_string(),
            token_type: TokenType::AccessToken,
            scopes: vec!["read".to_string(), "write".to_string()],
            claims: HashMap::new(),
            custom_lifetime: None,
        };

        let result = provider.issue_token(request).await;
        assert!(result.is_ok());

        let token = result.unwrap();
        assert_eq!(token.agent_id, "test_agent");
        assert_eq!(token.token_type, TokenType::AccessToken);
        assert_eq!(token.scopes, vec!["read", "write"]);
        assert!(!token.revoked);
    }

    #[tokio::test]
    async fn test_token_validation() {
        let mut provider = TokenProvider::new("test".to_string());
        
        let request = TokenIssuanceRequest {
            agent_id: "test_agent".to_string(),
            token_type: TokenType::AccessToken,
            scopes: vec!["read".to_string()],
            claims: HashMap::new(),
            custom_lifetime: None,
        };

        let token = provider.issue_token(request).await.unwrap();
        
        let result = provider.validate_token(&token.token_data).await;
        assert!(result.is_ok());

        let validation = result.unwrap();
        assert!(validation.valid);
        assert_eq!(validation.agent_id, "test_agent");
        assert_eq!(validation.scopes, vec!["read"]);
    }

    #[tokio::test]
    async fn test_token_refresh() {
        let mut provider = TokenProvider::new("test".to_string());
        
        let refresh_token = provider.issue_refresh_token("test_agent").await.unwrap();
        
        let result = provider.refresh_token(&refresh_token.token_data).await;
        assert!(result.is_ok());

        let new_token = result.unwrap();
        assert_eq!(new_token.agent_id, "test_agent");
        assert_eq!(new_token.token_type, TokenType::AccessToken);
    }

    #[tokio::test]
    async fn test_token_revocation() {
        let mut provider = TokenProvider::new("test".to_string());
        
        let request = TokenIssuanceRequest {
            agent_id: "test_agent".to_string(),
            token_type: TokenType::AccessToken,
            scopes: vec!["read".to_string()],
            claims: HashMap::new(),
            custom_lifetime: None,
        };

        let token = provider.issue_token(request).await.unwrap();
        
        // Revoke token
        let result = provider.revoke_token(&token.token_id).await;
        assert!(result.is_ok());

        // Validate revoked token
        let validation = provider.validate_token(&token.token_data).await.unwrap();
        assert!(!validation.valid);
        assert!(validation.error_message.is_some());
    }

    #[tokio::test]
    async fn test_token_listing() {
        let mut provider = TokenProvider::new("test".to_string());
        
        // Issue multiple tokens
        for i in 0..3 {
            let request = TokenIssuanceRequest {
                agent_id: format!("agent_{}", i),
                token_type: TokenType::AccessToken,
                scopes: vec!["read".to_string()],
                claims: HashMap::new(),
                custom_lifetime: None,
            };
            provider.issue_token(request).await.unwrap();
        }

        let tokens = provider.list_tokens("agent_0").await.unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].agent_id, "agent_0");
    }

    #[tokio::test]
    async fn test_token_cleanup() {
        let mut provider = TokenProvider::new("test".to_string());
        
        // Issue token with short lifetime
        let request = TokenIssuanceRequest {
            agent_id: "test_agent".to_string(),
            token_type: TokenType::AccessToken,
            scopes: vec!["read".to_string()],
            claims: HashMap::new(),
            custom_lifetime: Some(chrono::Duration::milliseconds(1)),
        };

        let token = provider.issue_token(request).await.unwrap();
        
        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        // Cleanup
        provider.cleanup_expired_tokens().await.unwrap();
        
        // Token should be removed
        let result = provider.get_token_info(&token.token_id).await;
        assert!(result.is_err());
    }
}