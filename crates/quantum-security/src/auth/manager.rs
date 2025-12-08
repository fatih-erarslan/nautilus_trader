//! Authentication Manager
//!
//! This module provides centralized authentication management for the quantum security system.

use crate::error::QuantumSecurityError;
use crate::types::*;
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Authentication Manager
#[derive(Debug, Clone)]
pub struct AuthenticationManager {
    pub id: Uuid,
    pub name: String,
    pub contexts: Arc<RwLock<HashMap<Uuid, AuthenticationContext>>>,
    pub challenges: Arc<RwLock<HashMap<Uuid, MFAChallenge>>>,
    pub policies: Arc<RwLock<HashMap<String, AuthenticationPolicy>>>,
    pub metrics: Arc<RwLock<AuthenticationMetrics>>,
    pub enabled: bool,
}

/// Authentication Session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationSession {
    pub session_id: Uuid,
    pub agent_id: String,
    pub context_id: Uuid,
    pub authentication_level: AuthenticationLevel,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub properties: HashMap<String, String>,
}

impl AuthenticationManager {
    /// Create new authentication manager
    pub fn new(name: String) -> Self {
        let mut policies = HashMap::new();
        policies.insert("default".to_string(), AuthenticationPolicy::default_policy());
        policies.insert("high_security".to_string(), AuthenticationPolicy::high_security_policy());

        Self {
            id: Uuid::new_v4(),
            name,
            contexts: Arc::new(RwLock::new(HashMap::new())),
            challenges: Arc::new(RwLock::new(HashMap::new())),
            policies: Arc::new(RwLock::new(policies)),
            metrics: Arc::new(RwLock::new(AuthenticationMetrics::default())),
            enabled: true,
        }
    }

    /// Start authentication process
    pub async fn start_authentication(
        &self,
        agent_id: String,
        policy_id: Option<String>,
    ) -> Result<AuthenticationContext, QuantumSecurityError> {
        // Get policy
        let policies = self.policies.read().await;
        let policy = policies.get(&policy_id.unwrap_or_else(|| "default".to_string()))
            .ok_or_else(|| QuantumSecurityError::PolicyNotFound("Policy not found".to_string()))?;

        // Create authentication context
        let context = AuthenticationContext::new(
            agent_id.clone(),
            policy.required_methods.clone(),
            policy.session_timeout,
        );

        // Store context
        let mut contexts = self.contexts.write().await;
        contexts.insert(context.context_id, context.clone());

        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.total_attempts += 1;

        Ok(context)
    }

    /// Create MFA challenge
    pub async fn create_challenge(
        &self,
        context_id: Uuid,
        method: AuthenticationMethod,
    ) -> Result<MFAChallenge, QuantumSecurityError> {
        // Verify context exists
        let contexts = self.contexts.read().await;
        let context = contexts.get(&context_id)
            .ok_or_else(|| QuantumSecurityError::ContextNotFound("Context not found".to_string()))?;

        if context.is_expired() {
            return Err(QuantumSecurityError::ContextExpired("Context has expired".to_string()));
        }

        // Create challenge data based on method
        let challenge_data = match method {
            AuthenticationMethod::Password => ChallengeData::Password {
                salt: vec![0u8; 32],
                iterations: 100_000,
                algorithm: "PBKDF2-SHA256".to_string(),
            },
            AuthenticationMethod::OneTimePassword => ChallengeData::TOTP {
                secret_id: format!("secret_{}", context_id),
                window_size: 1,
            },
            AuthenticationMethod::Biometric(bio_type) => ChallengeData::Biometric {
                biometric_type: bio_type,
                template_id: format!("template_{}", context_id),
                quality_threshold: 0.8,
            },
            AuthenticationMethod::HardwareKey => ChallengeData::HardwareKey {
                challenge: vec![0u8; 32],
                key_handle: vec![0u8; 64],
                app_id: "ats-cp-trader".to_string(),
            },
            AuthenticationMethod::QuantumEntanglement => ChallengeData::QuantumEntanglement {
                entanglement_id: format!("entanglement_{}", context_id),
                measurement_basis: vec![0u8; 256],
                expected_correlation: 0.95,
            },
            AuthenticationMethod::SmartCard => ChallengeData::SmartCard {
                challenge: vec![0u8; 32],
                certificate_id: format!("cert_{}", context_id),
            },
            _ => return Err(QuantumSecurityError::UnsupportedMethod("Unsupported method".to_string())),
        };

        // Create challenge
        let challenge = MFAChallenge::new(
            context_id,
            method,
            challenge_data,
            3,
            chrono::Duration::minutes(5),
        );

        // Store challenge
        let mut challenges = self.challenges.write().await;
        challenges.insert(challenge.challenge_id, challenge.clone());

        Ok(challenge)
    }

    /// Verify authentication response
    pub async fn verify_response(
        &self,
        challenge_id: Uuid,
        response: AuthenticationResponse,
    ) -> Result<AuthenticationResult, QuantumSecurityError> {
        // Get challenge
        let mut challenges = self.challenges.write().await;
        let challenge = challenges.get_mut(&challenge_id)
            .ok_or_else(|| QuantumSecurityError::ChallengeNotFound("Challenge not found".to_string()))?;

        if !challenge.can_attempt() {
            return Err(QuantumSecurityError::ChallengeExpired("Challenge expired or exhausted".to_string()));
        }

        challenge.increment_attempts();

        // Verify response (simplified for demo)
        let success = match (&challenge.challenge_data, &response.response_data) {
            (ChallengeData::Password { .. }, ResponseData::Password { .. }) => true,
            (ChallengeData::TOTP { .. }, ResponseData::TOTP { .. }) => true,
            (ChallengeData::Biometric { .. }, ResponseData::Biometric { .. }) => true,
            (ChallengeData::HardwareKey { .. }, ResponseData::HardwareKey { .. }) => true,
            (ChallengeData::QuantumEntanglement { .. }, ResponseData::QuantumEntanglement { .. }) => true,
            (ChallengeData::SmartCard { .. }, ResponseData::SmartCard { .. }) => true,
            _ => false,
        };

        let result = AuthenticationResult {
            success,
            method: challenge.method.clone(),
            confidence_score: if success { 0.95 } else { 0.0 },
            timestamp: chrono::Utc::now(),
            error_message: if success { None } else { Some("Authentication failed".to_string()) },
            metadata: HashMap::new(),
            quantum_signature: None,
        };

        if success {
            // Update context
            let mut contexts = self.contexts.write().await;
            if let Some(context) = contexts.get_mut(&challenge.context_id) {
                context.add_completed_method(challenge.method.clone());
            }

            // Update metrics
            let mut metrics = self.metrics.write().await;
            metrics.successful_attempts += 1;
            *metrics.method_usage.entry(challenge.method.clone()).or_insert(0) += 1;
        } else {
            let mut metrics = self.metrics.write().await;
            metrics.failed_attempts += 1;
        }

        Ok(result)
    }

    /// Get authentication context
    pub async fn get_context(&self, context_id: Uuid) -> Result<AuthenticationContext, QuantumSecurityError> {
        let contexts = self.contexts.read().await;
        contexts.get(&context_id)
            .cloned()
            .ok_or_else(|| QuantumSecurityError::ContextNotFound("Context not found".to_string()))
    }

    /// Create authentication session
    pub async fn create_session(
        &self,
        context_id: Uuid,
    ) -> Result<AuthenticationSession, QuantumSecurityError> {
        let contexts = self.contexts.read().await;
        let context = contexts.get(&context_id)
            .ok_or_else(|| QuantumSecurityError::ContextNotFound("Context not found".to_string()))?;

        if !context.is_complete() {
            return Err(QuantumSecurityError::IncompleteAuthentication("Authentication not complete".to_string()));
        }

        let session = AuthenticationSession {
            session_id: Uuid::new_v4(),
            agent_id: context.agent_id.clone(),
            context_id,
            authentication_level: context.authentication_level.clone(),
            created_at: chrono::Utc::now(),
            expires_at: context.expires_at,
            last_activity: chrono::Utc::now(),
            properties: HashMap::new(),
        };

        Ok(session)
    }

    /// Get authentication metrics
    pub async fn get_metrics(&self) -> Result<AuthenticationMetrics, QuantumSecurityError> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }

    /// Add authentication policy
    pub async fn add_policy(&self, policy: AuthenticationPolicy) -> Result<(), QuantumSecurityError> {
        let mut policies = self.policies.write().await;
        policies.insert(policy.policy_id.clone(), policy);
        Ok(())
    }

    /// Clean up expired contexts and challenges
    pub async fn cleanup_expired(&self) -> Result<(), QuantumSecurityError> {
        let now = chrono::Utc::now();

        // Clean up contexts
        let mut contexts = self.contexts.write().await;
        contexts.retain(|_, context| !context.is_expired());

        // Clean up challenges
        let mut challenges = self.challenges.write().await;
        challenges.retain(|_, challenge| !challenge.is_expired());

        Ok(())
    }

    /// Update authentication metrics
    pub async fn update_metrics(&self) -> Result<(), QuantumSecurityError> {
        let mut metrics = self.metrics.write().await;
        
        if metrics.total_attempts > 0 {
            metrics.success_rate = metrics.successful_attempts as f64 / metrics.total_attempts as f64;
        }

        Ok(())
    }
}

impl Default for AuthenticationManager {
    fn default() -> Self {
        Self::new("default-auth-manager".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_authentication_manager_creation() {
        let manager = AuthenticationManager::new("test".to_string());
        assert_eq!(manager.name, "test");
        assert!(manager.enabled);
    }

    #[tokio::test]
    async fn test_start_authentication() {
        let manager = AuthenticationManager::new("test".to_string());
        let result = manager.start_authentication("test_agent".to_string(), None).await;
        assert!(result.is_ok());
        
        let context = result.unwrap();
        assert_eq!(context.agent_id, "test_agent");
        assert!(!context.is_complete());
    }

    #[tokio::test]
    async fn test_create_challenge() {
        let manager = AuthenticationManager::new("test".to_string());
        let context = manager.start_authentication("test_agent".to_string(), None).await.unwrap();
        
        let challenge = manager.create_challenge(context.context_id, AuthenticationMethod::Password).await;
        assert!(challenge.is_ok());
        
        let challenge = challenge.unwrap();
        assert_eq!(challenge.method, AuthenticationMethod::Password);
        assert!(challenge.can_attempt());
    }

    #[tokio::test]
    async fn test_authentication_flow() {
        let manager = AuthenticationManager::new("test".to_string());
        
        // Start authentication
        let context = manager.start_authentication("test_agent".to_string(), None).await.unwrap();
        
        // Create password challenge
        let challenge = manager.create_challenge(context.context_id, AuthenticationMethod::Password).await.unwrap();
        
        // Submit response
        let response = AuthenticationResponse {
            challenge_id: challenge.challenge_id,
            response_data: ResponseData::Password {
                password_hash: vec![0u8; 32],
            },
            client_data: None,
            timestamp: chrono::Utc::now(),
        };
        
        let result = manager.verify_response(challenge.challenge_id, response).await;
        assert!(result.is_ok());
        
        let auth_result = result.unwrap();
        assert!(auth_result.success);
        assert_eq!(auth_result.method, AuthenticationMethod::Password);
    }

    #[tokio::test]
    async fn test_metrics_update() {
        let manager = AuthenticationManager::new("test".to_string());
        
        // Start authentication to generate metrics
        let _context = manager.start_authentication("test_agent".to_string(), None).await.unwrap();
        
        let metrics = manager.get_metrics().await.unwrap();
        assert_eq!(metrics.total_attempts, 1);
        assert_eq!(metrics.successful_attempts, 0);
    }
}