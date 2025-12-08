//! Quantum-Safe Authentication System
//!
//! This module implements multi-factor authentication with quantum resistance
//! for the ATS-CP trading system.

pub mod manager;
pub mod mfa;
pub mod biometric;
pub mod certificates;
pub mod tokens;

pub use manager::*;
pub use mfa::*;
pub use biometric::*;
pub use certificates::*;
pub use tokens::*;

use crate::error::QuantumSecurityError;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Authentication Methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuthenticationMethod {
    /// Password-based authentication
    Password,
    /// Hardware security key (FIDO2/WebAuthn)
    HardwareKey,
    /// Biometric authentication
    Biometric(BiometricType),
    /// Smart card authentication
    SmartCard,
    /// One-time password (TOTP/HOTP)
    OneTimePassword,
    /// Post-quantum digital certificates
    QuantumCertificate,
    /// Behavioral biometrics
    BehavioralBiometric,
    /// Quantum entanglement verification
    QuantumEntanglement,
}

/// Biometric Types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BiometricType {
    Fingerprint,
    FaceRecognition,
    IrisRecognition,
    VoiceRecognition,
    HandGeometry,
    Keystroke,
    Gait,
    Heartbeat,
}

/// Authentication Context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationContext {
    pub context_id: Uuid,
    pub agent_id: String,
    pub session_id: Option<Uuid>,
    pub authentication_level: AuthenticationLevel,
    pub completed_methods: Vec<AuthenticationMethod>,
    pub required_methods: Vec<AuthenticationMethod>,
    pub risk_score: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub source_ip: Option<String>,
    pub device_fingerprint: Option<String>,
    pub location: Option<GeoLocation>,
    pub quantum_verified: bool,
}

/// Authentication Level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuthenticationLevel {
    /// Basic authentication (single factor)
    Basic,
    /// Two-factor authentication
    TwoFactor,
    /// Multi-factor authentication (3+ factors)
    MultiFactor,
    /// High assurance authentication
    HighAssurance,
    /// Quantum-verified authentication
    QuantumVerified,
}

/// Geographic Location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeoLocation {
    pub country: String,
    pub region: String,
    pub city: String,
    pub latitude: f64,
    pub longitude: f64,
    pub accuracy_meters: f64,
}

/// Authentication Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResult {
    pub success: bool,
    pub method: AuthenticationMethod,
    pub confidence_score: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
    pub quantum_signature: Option<Vec<u8>>,
}

/// Authentication Policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationPolicy {
    pub policy_id: String,
    pub name: String,
    pub description: String,
    pub required_level: AuthenticationLevel,
    pub required_methods: Vec<AuthenticationMethod>,
    pub alternative_methods: Vec<Vec<AuthenticationMethod>>,
    pub max_risk_score: f64,
    pub session_timeout: chrono::Duration,
    pub quantum_verification_required: bool,
    pub applies_to: PolicyScope,
}

/// Policy Scope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyScope {
    AllAgents,
    AgentGroups(Vec<String>),
    SpecificAgents(Vec<String>),
    Operations(Vec<String>),
    Conditional(Vec<PolicyCondition>),
}

/// Policy Condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCondition {
    pub condition_type: PolicyConditionType,
    pub operator: ComparisonOperator,
    pub value: String,
}

/// Policy Condition Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyConditionType {
    RiskScore,
    Location,
    TimeOfDay,
    DeviceType,
    NetworkLocation,
    OperationType,
    TransactionAmount,
}

/// Comparison Operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    NotContains,
}

/// Multi-Factor Authentication Challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MFAChallenge {
    pub challenge_id: Uuid,
    pub context_id: Uuid,
    pub method: AuthenticationMethod,
    pub challenge_data: ChallengeData,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub attempts: u32,
    pub max_attempts: u32,
}

/// Challenge Data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChallengeData {
    /// Password challenge
    Password {
        salt: Vec<u8>,
        iterations: u32,
        algorithm: String,
    },
    /// TOTP challenge
    TOTP {
        secret_id: String,
        window_size: u32,
    },
    /// Biometric challenge
    Biometric {
        biometric_type: BiometricType,
        template_id: String,
        quality_threshold: f64,
    },
    /// Hardware key challenge (FIDO2)
    HardwareKey {
        challenge: Vec<u8>,
        key_handle: Vec<u8>,
        app_id: String,
    },
    /// Quantum entanglement challenge
    QuantumEntanglement {
        entanglement_id: String,
        measurement_basis: Vec<u8>,
        expected_correlation: f64,
    },
    /// Smart card challenge
    SmartCard {
        challenge: Vec<u8>,
        certificate_id: String,
    },
}

/// Authentication Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResponse {
    pub challenge_id: Uuid,
    pub response_data: ResponseData,
    pub client_data: Option<HashMap<String, String>>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Response Data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseData {
    /// Password response
    Password {
        password_hash: Vec<u8>,
    },
    /// TOTP response
    TOTP {
        code: String,
    },
    /// Biometric response
    Biometric {
        biometric_data: Vec<u8>,
        quality_score: f64,
    },
    /// Hardware key response
    HardwareKey {
        signature: Vec<u8>,
        counter: u32,
        user_presence: bool,
    },
    /// Quantum entanglement response
    QuantumEntanglement {
        measurement_results: Vec<bool>,
        correlation_coefficient: f64,
    },
    /// Smart card response
    SmartCard {
        signature: Vec<u8>,
        certificate: Vec<u8>,
    },
}

/// Authentication Metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuthenticationMetrics {
    pub total_attempts: u64,
    pub successful_attempts: u64,
    pub failed_attempts: u64,
    pub success_rate: f64,
    pub average_authentication_time_ms: f64,
    pub method_usage: HashMap<AuthenticationMethod, u64>,
    pub risk_score_distribution: HashMap<String, u64>, // "low", "medium", "high"
    pub quantum_verification_count: u64,
    pub policy_violations: u64,
    pub suspicious_activities: u64,
}

/// Authentication Event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationEvent {
    pub event_id: Uuid,
    pub event_type: AuthenticationEventType,
    pub agent_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context_id: Option<Uuid>,
    pub method: Option<AuthenticationMethod>,
    pub success: bool,
    pub risk_score: f64,
    pub metadata: HashMap<String, String>,
}

/// Authentication Event Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthenticationEventType {
    AuthenticationStarted,
    AuthenticationCompleted,
    AuthenticationFailed,
    MFAChallengeIssued,
    MFAChallengeCompleted,
    MFAChallengeFailed,
    RiskScoreCalculated,
    PolicyViolation,
    SuspiciousActivity,
    BiometricEnrollment,
    BiometricUpdate,
    CertificateIssued,
    CertificateRevoked,
    QuantumVerificationCompleted,
}

impl AuthenticationContext {
    /// Create new authentication context
    pub fn new(
        agent_id: String,
        required_methods: Vec<AuthenticationMethod>,
        session_timeout: chrono::Duration,
    ) -> Self {
        let now = chrono::Utc::now();
        
        Self {
            context_id: Uuid::new_v4(),
            agent_id,
            session_id: None,
            authentication_level: AuthenticationLevel::Basic,
            completed_methods: Vec::new(),
            required_methods,
            risk_score: 0.0,
            created_at: now,
            expires_at: now + session_timeout,
            source_ip: None,
            device_fingerprint: None,
            location: None,
            quantum_verified: false,
        }
    }
    
    /// Add completed authentication method
    pub fn add_completed_method(&mut self, method: AuthenticationMethod) {
        if !self.completed_methods.contains(&method) {
            self.completed_methods.push(method);
            self.update_authentication_level();
        }
    }
    
    /// Check if authentication is complete
    pub fn is_complete(&self) -> bool {
        self.required_methods.iter().all(|method| self.completed_methods.contains(method))
    }
    
    /// Check if context is expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expires_at
    }
    
    /// Update authentication level based on completed methods
    fn update_authentication_level(&mut self) {
        let method_count = self.completed_methods.len();
        let has_quantum = self.completed_methods.iter()
            .any(|m| matches!(m, AuthenticationMethod::QuantumCertificate | AuthenticationMethod::QuantumEntanglement));
        
        self.authentication_level = match (method_count, has_quantum, self.quantum_verified) {
            (_, _, true) => AuthenticationLevel::QuantumVerified,
            (count, true, _) if count >= 3 => AuthenticationLevel::HighAssurance,
            (count, _, _) if count >= 3 => AuthenticationLevel::MultiFactor,
            (2, _, _) => AuthenticationLevel::TwoFactor,
            (1, _, _) => AuthenticationLevel::Basic,
            _ => AuthenticationLevel::Basic,
        };
    }
    
    /// Set quantum verification status
    pub fn set_quantum_verified(&mut self, verified: bool) {
        self.quantum_verified = verified;
        if verified {
            self.authentication_level = AuthenticationLevel::QuantumVerified;
        }
    }
    
    /// Update risk score
    pub fn update_risk_score(&mut self, score: f64) {
        self.risk_score = score.clamp(0.0, 1.0);
    }
    
    /// Get remaining time until expiry
    pub fn remaining_time(&self) -> chrono::Duration {
        self.expires_at.signed_duration_since(chrono::Utc::now())
    }
}

impl MFAChallenge {
    /// Create new MFA challenge
    pub fn new(
        context_id: Uuid,
        method: AuthenticationMethod,
        challenge_data: ChallengeData,
        max_attempts: u32,
        expires_in: chrono::Duration,
    ) -> Self {
        let now = chrono::Utc::now();
        
        Self {
            challenge_id: Uuid::new_v4(),
            context_id,
            method,
            challenge_data,
            created_at: now,
            expires_at: now + expires_in,
            attempts: 0,
            max_attempts,
        }
    }
    
    /// Check if challenge is expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expires_at
    }
    
    /// Check if max attempts reached
    pub fn is_exhausted(&self) -> bool {
        self.attempts >= self.max_attempts
    }
    
    /// Increment attempt counter
    pub fn increment_attempts(&mut self) {
        self.attempts += 1;
    }
    
    /// Check if challenge can be attempted
    pub fn can_attempt(&self) -> bool {
        !self.is_expired() && !self.is_exhausted()
    }
}

impl AuthenticationPolicy {
    /// Create default authentication policy
    pub fn default_policy() -> Self {
        Self {
            policy_id: "default".to_string(),
            name: "Default Authentication Policy".to_string(),
            description: "Standard multi-factor authentication policy".to_string(),
            required_level: AuthenticationLevel::TwoFactor,
            required_methods: vec![
                AuthenticationMethod::Password,
                AuthenticationMethod::OneTimePassword,
            ],
            alternative_methods: vec![
                vec![
                    AuthenticationMethod::HardwareKey,
                    AuthenticationMethod::Biometric(BiometricType::Fingerprint),
                ],
            ],
            max_risk_score: 0.7,
            session_timeout: chrono::Duration::hours(8),
            quantum_verification_required: false,
            applies_to: PolicyScope::AllAgents,
        }
    }
    
    /// Create high-security policy
    pub fn high_security_policy() -> Self {
        Self {
            policy_id: "high_security".to_string(),
            name: "High Security Authentication Policy".to_string(),
            description: "High assurance authentication with quantum verification".to_string(),
            required_level: AuthenticationLevel::QuantumVerified,
            required_methods: vec![
                AuthenticationMethod::Password,
                AuthenticationMethod::HardwareKey,
                AuthenticationMethod::Biometric(BiometricType::Fingerprint),
                AuthenticationMethod::QuantumCertificate,
            ],
            alternative_methods: vec![],
            max_risk_score: 0.3,
            session_timeout: chrono::Duration::hours(2),
            quantum_verification_required: true,
            applies_to: PolicyScope::AllAgents,
        }
    }
    
    /// Check if policy applies to agent
    pub fn applies_to_agent(&self, agent_id: &str) -> bool {
        match &self.applies_to {
            PolicyScope::AllAgents => true,
            PolicyScope::SpecificAgents(agents) => agents.contains(&agent_id.to_string()),
            PolicyScope::AgentGroups(_) => {
                // Implementation would check group membership
                false
            },
            PolicyScope::Operations(_) => {
                // Implementation would check operation context
                false
            },
            PolicyScope::Conditional(_) => {
                // Implementation would evaluate conditions
                false
            },
        }
    }
    
    /// Check if authentication level meets policy requirements
    pub fn meets_requirements(&self, level: &AuthenticationLevel, methods: &[AuthenticationMethod]) -> bool {
        // Check authentication level
        if level < &self.required_level {
            return false;
        }
        
        // Check required methods
        if !self.required_methods.iter().all(|method| methods.contains(method)) {
            // Check alternative methods
            if !self.alternative_methods.iter().any(|alt_methods| {
                alt_methods.iter().all(|method| methods.contains(method))
            }) {
                return false;
            }
        }
        
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_authentication_context_creation() {
        let context = AuthenticationContext::new(
            "test_agent".to_string(),
            vec![AuthenticationMethod::Password, AuthenticationMethod::OneTimePassword],
            chrono::Duration::hours(8),
        );
        
        assert_eq!(context.agent_id, "test_agent");
        assert_eq!(context.authentication_level, AuthenticationLevel::Basic);
        assert!(!context.is_complete());
        assert!(!context.is_expired());
    }
    
    #[test]
    fn test_authentication_context_completion() {
        let mut context = AuthenticationContext::new(
            "test_agent".to_string(),
            vec![AuthenticationMethod::Password, AuthenticationMethod::OneTimePassword],
            chrono::Duration::hours(8),
        );
        
        context.add_completed_method(AuthenticationMethod::Password);
        assert_eq!(context.authentication_level, AuthenticationLevel::Basic);
        assert!(!context.is_complete());
        
        context.add_completed_method(AuthenticationMethod::OneTimePassword);
        assert_eq!(context.authentication_level, AuthenticationLevel::TwoFactor);
        assert!(context.is_complete());
    }
    
    #[test]
    fn test_mfa_challenge() {
        let challenge = MFAChallenge::new(
            Uuid::new_v4(),
            AuthenticationMethod::TOTP,
            ChallengeData::TOTP {
                secret_id: "test_secret".to_string(),
                window_size: 1,
            },
            3,
            chrono::Duration::minutes(5),
        );
        
        assert!(!challenge.is_expired());
        assert!(!challenge.is_exhausted());
        assert!(challenge.can_attempt());
        assert_eq!(challenge.attempts, 0);
        assert_eq!(challenge.max_attempts, 3);
    }
    
    #[test]
    fn test_authentication_policy() {
        let policy = AuthenticationPolicy::default_policy();
        
        assert!(policy.applies_to_agent("any_agent"));
        assert_eq!(policy.required_level, AuthenticationLevel::TwoFactor);
        
        let methods = vec![AuthenticationMethod::Password, AuthenticationMethod::OneTimePassword];
        assert!(policy.meets_requirements(&AuthenticationLevel::TwoFactor, &methods));
        
        let insufficient_methods = vec![AuthenticationMethod::Password];
        assert!(!policy.meets_requirements(&AuthenticationLevel::Basic, &insufficient_methods));
    }
    
    #[test]
    fn test_quantum_verification() {
        let mut context = AuthenticationContext::new(
            "test_agent".to_string(),
            vec![AuthenticationMethod::QuantumCertificate],
            chrono::Duration::hours(8),
        );
        
        context.add_completed_method(AuthenticationMethod::QuantumCertificate);
        context.set_quantum_verified(true);
        
        assert_eq!(context.authentication_level, AuthenticationLevel::QuantumVerified);
        assert!(context.quantum_verified);
        assert!(context.is_complete());
    }
}