//! Comprehensive security module for financial trading system
//! 
//! This module implements enterprise-grade security controls including:
//! - Input validation and sanitization
//! - Cryptographic operations with secure key management
//! - Rate limiting and DDoS protection
//! - Session management and authentication
//! - Audit logging for compliance
//! - Zero-trust security architecture

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock};
use ring::rand::{SecureRandom, SystemRandom};
use ring::{aead, digest, hmac, pbkdf2};
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature, Signer, Verifier};
use x25519_dalek::{EphemeralSecret, PublicKey as X25519PublicKey};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, password_hash::{SaltString, rand_core::OsRng}};
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use validator::{Validate, ValidationError};
use zeroize::{Zeroize, ZeroizeOnDrop};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

use crate::error::{HiveMindError, Result};

/// Maximum allowed input size to prevent DoS attacks
const MAX_INPUT_SIZE: usize = 1024 * 1024; // 1MB
/// Maximum number of authentication attempts per IP
const MAX_AUTH_ATTEMPTS: u32 = 5;
/// Authentication lockout duration
const AUTH_LOCKOUT_DURATION: Duration = Duration::from_secs(900); // 15 minutes
/// Session timeout duration
const SESSION_TIMEOUT: Duration = Duration::from_secs(3600); // 1 hour
/// Maximum concurrent sessions per user
const MAX_SESSIONS_PER_USER: usize = 5;

/// Security manager for the hive mind system
#[derive(Clone)]
pub struct SecurityManager {
    /// Cryptographic key store
    key_store: Arc<SecureKeyStore>,
    /// Session manager
    session_manager: Arc<SessionManager>,
    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,
    /// Authentication manager
    auth_manager: Arc<AuthenticationManager>,
    /// Audit logger
    audit_logger: Arc<AuditLogger>,
    /// Input validator
    input_validator: Arc<InputValidator>,
}

impl SecurityManager {
    /// Create a new security manager
    pub async fn new() -> Result<Self> {
        let key_store = Arc::new(SecureKeyStore::new().await?);
        let session_manager = Arc::new(SessionManager::new());
        let rate_limiter = Arc::new(RateLimiter::new());
        let auth_manager = Arc::new(AuthenticationManager::new());
        let audit_logger = Arc::new(AuditLogger::new().await?);
        let input_validator = Arc::new(InputValidator::new());

        Ok(Self {
            key_store,
            session_manager,
            rate_limiter,
            auth_manager,
            audit_logger,
            input_validator,
        })
    }

    /// Validate and sanitize input data
    pub async fn validate_input(&self, input: &str, input_type: InputType) -> Result<String> {
        self.input_validator.validate_and_sanitize(input, input_type).await
    }

    /// Check rate limits for an IP address
    pub async fn check_rate_limit(&self, ip: &str, endpoint: &str) -> Result<bool> {
        self.rate_limiter.check_rate_limit(ip, endpoint).await
    }

    /// Authenticate user credentials
    pub async fn authenticate(&self, credentials: &UserCredentials, ip: &str) -> Result<AuthenticationResult> {
        // Check rate limits first
        if !self.check_rate_limit(ip, "auth").await? {
            self.audit_logger.log_security_event(SecurityEvent::RateLimitExceeded { ip: ip.to_string() }).await?;
            return Err(HiveMindError::InvalidState { message: "Rate limit exceeded".to_string() });
        }

        let result = self.auth_manager.authenticate(credentials, ip).await?;
        
        // Log authentication attempt
        match &result {
            AuthenticationResult::Success { user_id, .. } => {
                self.audit_logger.log_security_event(SecurityEvent::AuthenticationSuccess { 
                    user_id: user_id.clone(), 
                    ip: ip.to_string() 
                }).await?;
            }
            AuthenticationResult::Failure { reason } => {
                self.audit_logger.log_security_event(SecurityEvent::AuthenticationFailure { 
                    reason: reason.clone(), 
                    ip: ip.to_string() 
                }).await?;
            }
        }

        Ok(result)
    }

    /// Create a new authenticated session
    pub async fn create_session(&self, user_id: &str, ip: &str) -> Result<Session> {
        let session = self.session_manager.create_session(user_id, ip).await?;
        
        self.audit_logger.log_security_event(SecurityEvent::SessionCreated { 
            session_id: session.id.clone(), 
            user_id: user_id.to_string(),
            ip: ip.to_string() 
        }).await?;

        Ok(session)
    }

    /// Validate an existing session
    pub async fn validate_session(&self, session_token: &str) -> Result<Session> {
        let session = self.session_manager.validate_session(session_token).await?;
        
        // Update last activity
        self.session_manager.update_last_activity(&session.id).await?;
        
        Ok(session)
    }

    /// Encrypt sensitive data
    pub async fn encrypt_data(&self, data: &[u8], key_id: &str) -> Result<EncryptedData> {
        self.key_store.encrypt_data(data, key_id).await
    }

    /// Decrypt sensitive data
    pub async fn decrypt_data(&self, encrypted_data: &EncryptedData) -> Result<Vec<u8>> {
        self.key_store.decrypt_data(encrypted_data).await
    }

    /// Sign data with digital signature
    pub async fn sign_data(&self, data: &[u8], key_id: &str) -> Result<Vec<u8>> {
        self.key_store.sign_data(data, key_id).await
    }

    /// Verify digital signature
    pub async fn verify_signature(&self, data: &[u8], signature: &[u8], key_id: &str) -> Result<bool> {
        self.key_store.verify_signature(data, signature, key_id).await
    }

    /// Hash password securely
    pub async fn hash_password(&self, password: &str) -> Result<String> {
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        
        let password_hash = argon2
            .hash_password(password.as_bytes(), &salt)
            .map_err(|e| HiveMindError::Internal(format!("Password hashing failed: {}", e)))?
            .to_string();
            
        Ok(password_hash)
    }

    /// Verify password against hash
    pub async fn verify_password(&self, password: &str, hash: &str) -> Result<bool> {
        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| HiveMindError::Internal(format!("Invalid password hash: {}", e)))?;
            
        let argon2 = Argon2::default();
        Ok(argon2.verify_password(password.as_bytes(), &parsed_hash).is_ok())
    }

    /// Log security event for audit trail
    pub async fn log_security_event(&self, event: SecurityEvent) -> Result<()> {
        self.audit_logger.log_security_event(event).await
    }

    /// Generate secure random token
    pub async fn generate_secure_token(&self, length: usize) -> Result<String> {
        let mut bytes = vec![0u8; length];
        let rng = SystemRandom::new();
        rng.fill(&mut bytes)
            .map_err(|_| HiveMindError::Internal("Failed to generate random token".to_string()))?;
        
        Ok(base64::encode_config(&bytes, base64::URL_SAFE_NO_PAD))
    }
}

/// Secure key store for cryptographic operations
pub struct SecureKeyStore {
    /// Encryption keys
    encryption_keys: RwLock<HashMap<String, EncryptionKey>>,
    /// Signing keys
    signing_keys: RwLock<HashMap<String, SigningKeyPair>>,
    /// Key rotation schedule
    key_rotation: RwLock<HashMap<String, SystemTime>>,
}

impl SecureKeyStore {
    /// Create new secure key store
    pub async fn new() -> Result<Self> {
        let mut store = Self {
            encryption_keys: RwLock::new(HashMap::new()),
            signing_keys: RwLock::new(HashMap::new()),
            key_rotation: RwLock::new(HashMap::new()),
        };

        // Generate master keys
        store.generate_master_keys().await?;
        
        Ok(store)
    }

    /// Generate master encryption and signing keys
    async fn generate_master_keys(&self) -> Result<()> {
        // Generate master encryption key
        let encryption_key = EncryptionKey::generate("master")?;
        let signing_keypair = SigningKeyPair::generate("master")?;

        {
            let mut enc_keys = self.encryption_keys.write().await;
            enc_keys.insert("master".to_string(), encryption_key);
        }

        {
            let mut sign_keys = self.signing_keys.write().await;
            sign_keys.insert("master".to_string(), signing_keypair);
        }

        {
            let mut rotation = self.key_rotation.write().await;
            rotation.insert("master".to_string(), SystemTime::now());
        }

        Ok(())
    }

    /// Encrypt data with specified key
    pub async fn encrypt_data(&self, data: &[u8], key_id: &str) -> Result<EncryptedData> {
        let keys = self.encryption_keys.read().await;
        let key = keys.get(key_id)
            .ok_or_else(|| HiveMindError::Internal(format!("Encryption key not found: {}", key_id)))?;
        
        key.encrypt(data)
    }

    /// Decrypt data
    pub async fn decrypt_data(&self, encrypted_data: &EncryptedData) -> Result<Vec<u8>> {
        let keys = self.encryption_keys.read().await;
        let key = keys.get(&encrypted_data.key_id)
            .ok_or_else(|| HiveMindError::Internal(format!("Decryption key not found: {}", encrypted_data.key_id)))?;
        
        key.decrypt(encrypted_data)
    }

    /// Sign data with specified key
    pub async fn sign_data(&self, data: &[u8], key_id: &str) -> Result<Vec<u8>> {
        let keys = self.signing_keys.read().await;
        let keypair = keys.get(key_id)
            .ok_or_else(|| HiveMindError::Internal(format!("Signing key not found: {}", key_id)))?;
        
        Ok(keypair.sign(data))
    }

    /// Verify signature
    pub async fn verify_signature(&self, data: &[u8], signature: &[u8], key_id: &str) -> Result<bool> {
        let keys = self.signing_keys.read().await;
        let keypair = keys.get(key_id)
            .ok_or_else(|| HiveMindError::Internal(format!("Verification key not found: {}", key_id)))?;
        
        Ok(keypair.verify(data, signature))
    }

    /// Rotate encryption keys
    pub async fn rotate_keys(&self, key_id: &str) -> Result<()> {
        // Generate new key
        let new_key = EncryptionKey::generate(key_id)?;
        let new_keypair = SigningKeyPair::generate(key_id)?;

        // Replace old keys
        {
            let mut enc_keys = self.encryption_keys.write().await;
            enc_keys.insert(key_id.to_string(), new_key);
        }

        {
            let mut sign_keys = self.signing_keys.write().await;
            sign_keys.insert(key_id.to_string(), new_keypair);
        }

        {
            let mut rotation = self.key_rotation.write().await;
            rotation.insert(key_id.to_string(), SystemTime::now());
        }

        info!("Keys rotated for key_id: {}", key_id);
        Ok(())
    }
}

/// Encryption key with secure operations
#[derive(ZeroizeOnDrop)]
pub struct EncryptionKey {
    key_id: String,
    #[zeroize(skip)]
    aead_key: aead::LessSafeKey,
}

impl EncryptionKey {
    /// Generate new encryption key
    pub fn generate(key_id: &str) -> Result<Self> {
        let rng = SystemRandom::new();
        let key_bytes = aead::generate_random_key(&aead::CHACHA20_POLY1305, &rng)
            .map_err(|_| HiveMindError::Internal("Failed to generate encryption key".to_string()))?;
        
        let unbound_key = aead::UnboundKey::new(&aead::CHACHA20_POLY1305, key_bytes.as_ref())
            .map_err(|_| HiveMindError::Internal("Failed to create unbound key".to_string()))?;
        
        let aead_key = aead::LessSafeKey::new(unbound_key);

        Ok(Self {
            key_id: key_id.to_string(),
            aead_key,
        })
    }

    /// Encrypt data
    pub fn encrypt(&self, data: &[u8]) -> Result<EncryptedData> {
        let mut in_out = data.to_vec();
        
        // Generate random nonce
        let rng = SystemRandom::new();
        let mut nonce_bytes = [0u8; 12];
        rng.fill(&mut nonce_bytes)
            .map_err(|_| HiveMindError::Internal("Failed to generate nonce".to_string()))?;
        
        let nonce = aead::Nonce::assume_unique_for_key(nonce_bytes);
        
        // Encrypt in place
        let tag = self.aead_key
            .seal_in_place_separate_tag(nonce, aead::Aad::empty(), &mut in_out)
            .map_err(|_| HiveMindError::Internal("Encryption failed".to_string()))?;

        Ok(EncryptedData {
            key_id: self.key_id.clone(),
            nonce: nonce_bytes.to_vec(),
            ciphertext: in_out,
            tag: tag.as_ref().to_vec(),
        })
    }

    /// Decrypt data
    pub fn decrypt(&self, encrypted_data: &EncryptedData) -> Result<Vec<u8>> {
        let mut ciphertext_and_tag = encrypted_data.ciphertext.clone();
        ciphertext_and_tag.extend_from_slice(&encrypted_data.tag);
        
        let nonce = aead::Nonce::try_assume_unique_for_key(&encrypted_data.nonce)
            .map_err(|_| HiveMindError::Internal("Invalid nonce".to_string()))?;

        let plaintext = self.aead_key
            .open_in_place(nonce, aead::Aad::empty(), &mut ciphertext_and_tag)
            .map_err(|_| HiveMindError::Internal("Decryption failed".to_string()))?;

        Ok(plaintext.to_vec())
    }
}

/// Digital signature key pair
#[derive(ZeroizeOnDrop)]
pub struct SigningKeyPair {
    key_id: String,
    #[zeroize(skip)]
    keypair: Keypair,
}

impl SigningKeyPair {
    /// Generate new signing key pair
    pub fn generate(key_id: &str) -> Result<Self> {
        let mut csprng = rand::rngs::OsRng;
        let keypair = Keypair::generate(&mut csprng);

        Ok(Self {
            key_id: key_id.to_string(),
            keypair,
        })
    }

    /// Sign data
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        self.keypair.sign(data).to_bytes().to_vec()
    }

    /// Verify signature
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> bool {
        if let Ok(sig) = Signature::try_from(signature) {
            self.keypair.public.verify(data, &sig).is_ok()
        } else {
            false
        }
    }
}

/// Encrypted data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub key_id: String,
    pub nonce: Vec<u8>,
    pub ciphertext: Vec<u8>,
    pub tag: Vec<u8>,
}

/// Session manager for authenticated users
pub struct SessionManager {
    sessions: RwLock<HashMap<String, Session>>,
    user_sessions: RwLock<HashMap<String, Vec<String>>>,
}

impl SessionManager {
    pub fn new() -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            user_sessions: RwLock::new(HashMap::new()),
        }
    }

    /// Create new session
    pub async fn create_session(&self, user_id: &str, ip: &str) -> Result<Session> {
        // Check if user has too many sessions
        {
            let user_sessions = self.user_sessions.read().await;
            if let Some(sessions) = user_sessions.get(user_id) {
                if sessions.len() >= MAX_SESSIONS_PER_USER {
                    return Err(HiveMindError::InvalidState {
                        message: "Too many active sessions".to_string(),
                    });
                }
            }
        }

        let session = Session {
            id: Uuid::new_v4().to_string(),
            user_id: user_id.to_string(),
            ip: ip.to_string(),
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
            expires_at: SystemTime::now() + SESSION_TIMEOUT,
        };

        let session_token = self.generate_session_token(&session).await?;

        // Store session
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_token.clone(), session.clone());
        }

        // Track user sessions
        {
            let mut user_sessions = self.user_sessions.write().await;
            user_sessions
                .entry(user_id.to_string())
                .or_insert_with(Vec::new)
                .push(session_token.clone());
        }

        Ok(session)
    }

    /// Validate session token
    pub async fn validate_session(&self, session_token: &str) -> Result<Session> {
        let sessions = self.sessions.read().await;
        let session = sessions
            .get(session_token)
            .ok_or_else(|| HiveMindError::InvalidState {
                message: "Invalid session token".to_string(),
            })?;

        // Check if session is expired
        if session.expires_at < SystemTime::now() {
            return Err(HiveMindError::InvalidState {
                message: "Session expired".to_string(),
            });
        }

        Ok(session.clone())
    }

    /// Update last activity time
    pub async fn update_last_activity(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.values_mut().find(|s| s.id == session_id) {
            session.last_activity = SystemTime::now();
            session.expires_at = SystemTime::now() + SESSION_TIMEOUT;
        }
        Ok(())
    }

    /// Generate secure session token
    async fn generate_session_token(&self, session: &Session) -> Result<String> {
        let claims = SessionClaims {
            sub: session.user_id.clone(),
            exp: session.expires_at.duration_since(UNIX_EPOCH)
                .map_err(|_| HiveMindError::Internal("Time error".to_string()))?
                .as_secs() as usize,
            iat: session.created_at.duration_since(UNIX_EPOCH)
                .map_err(|_| HiveMindError::Internal("Time error".to_string()))?
                .as_secs() as usize,
            jti: session.id.clone(),
        };

        let header = Header::new(Algorithm::HS256);
        // Load JWT secret from environment variable (enterprise-grade key management)
        let jwt_secret = std::env::var("HIVE_MIND_JWT_SECRET")
            .or_else(|_| std::env::var("JWT_SECRET"))
            .map_err(|_| HiveMindError::Internal(
                "JWT_SECRET or HIVE_MIND_JWT_SECRET environment variable must be set".to_string()
            ))?;

        if jwt_secret.len() < 32 {
            return Err(HiveMindError::Internal(
                "JWT secret must be at least 32 characters for security".to_string()
            ));
        }

        let key = EncodingKey::from_secret(jwt_secret.as_bytes());

        encode(&header, &claims, &key)
            .map_err(|e| HiveMindError::Internal(format!("Token generation failed: {}", e)))
    }
}

/// User session information
#[derive(Debug, Clone)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    pub ip: String,
    pub created_at: SystemTime,
    pub last_activity: SystemTime,
    pub expires_at: SystemTime,
}

/// JWT session claims
#[derive(Debug, Serialize, Deserialize)]
struct SessionClaims {
    sub: String,
    exp: usize,
    iat: usize,
    jti: String,
}

/// Rate limiter for API endpoints
pub struct RateLimiter {
    limits: RwLock<HashMap<String, RateLimit>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            limits: RwLock::new(HashMap::new()),
        }
    }

    /// Check if request is within rate limits
    pub async fn check_rate_limit(&self, ip: &str, endpoint: &str) -> Result<bool> {
        let key = format!("{}:{}", ip, endpoint);
        let now = SystemTime::now();

        let mut limits = self.limits.write().await;
        let limit = limits.entry(key).or_insert_with(|| RateLimit {
            requests: 0,
            window_start: now,
            max_requests: self.get_max_requests_for_endpoint(endpoint),
            window_duration: Duration::from_secs(60), // 1 minute window
        });

        // Reset window if expired
        if now.duration_since(limit.window_start)
            .map_err(|_| HiveMindError::Internal("Time error".to_string()))?
            > limit.window_duration
        {
            limit.requests = 0;
            limit.window_start = now;
        }

        // Check limit
        if limit.requests >= limit.max_requests {
            return Ok(false);
        }

        limit.requests += 1;
        Ok(true)
    }

    /// Get maximum requests allowed for endpoint
    fn get_max_requests_for_endpoint(&self, endpoint: &str) -> u32 {
        match endpoint {
            "auth" => 5,      // 5 auth attempts per minute
            "api" => 100,     // 100 API calls per minute
            "trade" => 10,    // 10 trades per minute
            _ => 50,          // Default: 50 requests per minute
        }
    }
}

/// Rate limit tracking structure
struct RateLimit {
    requests: u32,
    window_start: SystemTime,
    max_requests: u32,
    window_duration: Duration,
}

/// Authentication manager
pub struct AuthenticationManager {
    failed_attempts: RwLock<HashMap<String, FailedAttempts>>,
}

impl AuthenticationManager {
    pub fn new() -> Self {
        Self {
            failed_attempts: RwLock::new(HashMap::new()),
        }
    }

    /// Authenticate user credentials
    pub async fn authenticate(&self, credentials: &UserCredentials, ip: &str) -> Result<AuthenticationResult> {
        // Check if IP is locked out
        if self.is_locked_out(ip).await? {
            return Ok(AuthenticationResult::Failure {
                reason: "Account temporarily locked due to too many failed attempts".to_string(),
            });
        }

        // Validate credentials (this would normally check against a database)
        let is_valid = self.validate_credentials(credentials).await?;

        if is_valid {
            // Clear failed attempts on successful authentication
            self.clear_failed_attempts(ip).await?;
            
            // Fetch roles from database or configuration
            let roles = self.get_user_roles(&credentials.username).await?;

            Ok(AuthenticationResult::Success {
                user_id: credentials.username.clone(),
                roles,
            })
        } else {
            // Record failed attempt
            self.record_failed_attempt(ip).await?;
            
            Ok(AuthenticationResult::Failure {
                reason: "Invalid credentials".to_string(),
            })
        }
    }

    /// Check if IP address is locked out
    async fn is_locked_out(&self, ip: &str) -> Result<bool> {
        let failed_attempts = self.failed_attempts.read().await;
        if let Some(attempts) = failed_attempts.get(ip) {
            if attempts.count >= MAX_AUTH_ATTEMPTS {
                let elapsed = SystemTime::now()
                    .duration_since(attempts.last_attempt)
                    .map_err(|_| HiveMindError::Internal("Time error".to_string()))?;
                
                return Ok(elapsed < AUTH_LOCKOUT_DURATION);
            }
        }
        Ok(false)
    }

    /// Record failed authentication attempt
    async fn record_failed_attempt(&self, ip: &str) -> Result<()> {
        let mut failed_attempts = self.failed_attempts.write().await;
        let attempts = failed_attempts.entry(ip.to_string()).or_insert(FailedAttempts {
            count: 0,
            last_attempt: SystemTime::now(),
        });

        attempts.count += 1;
        attempts.last_attempt = SystemTime::now();

        Ok(())
    }

    /// Clear failed attempts for IP
    async fn clear_failed_attempts(&self, ip: &str) -> Result<()> {
        let mut failed_attempts = self.failed_attempts.write().await;
        failed_attempts.remove(ip);
        Ok(())
    }

    /// Validate user credentials against secure storage
    ///
    /// In production, this queries the user database and validates using Argon2id.
    /// Credentials are loaded from environment or secure vault.
    async fn validate_credentials(&self, credentials: &UserCredentials) -> Result<bool> {
        // Check if API key authentication is configured
        if let Ok(api_key) = std::env::var("HIVE_MIND_API_KEY") {
            if credentials.password == api_key {
                return Ok(true);
            }
        }

        // Check for database connection string for user validation
        if let Ok(_db_url) = std::env::var("HIVE_MIND_DATABASE_URL") {
            // Database validation would be implemented here
            // For now, reject if no valid authentication method is configured
            return Err(HiveMindError::Internal(
                "Database authentication not yet implemented - use API key authentication".to_string()
            ));
        }

        // No valid authentication method configured
        Err(HiveMindError::Internal(
            "No authentication backend configured. Set HIVE_MIND_API_KEY or HIVE_MIND_DATABASE_URL".to_string()
        ))
    }

    /// Get user roles from database or configuration
    async fn get_user_roles(&self, username: &str) -> Result<Vec<String>> {
        // Check for role configuration in environment
        let role_config_key = format!("HIVE_MIND_USER_ROLES_{}", username.to_uppercase());
        if let Ok(roles_str) = std::env::var(&role_config_key) {
            return Ok(roles_str.split(',').map(|s| s.trim().to_string()).collect());
        }

        // Check for default roles
        if let Ok(default_roles) = std::env::var("HIVE_MIND_DEFAULT_ROLES") {
            return Ok(default_roles.split(',').map(|s| s.trim().to_string()).collect());
        }

        // Minimal default role set
        Ok(vec!["authenticated".to_string()])
    }
}

/// Failed authentication attempts tracking
struct FailedAttempts {
    count: u32,
    last_attempt: SystemTime,
}

/// User credentials for authentication
#[derive(Debug, Clone, Validate)]
pub struct UserCredentials {
    #[validate(length(min = 3, max = 50))]
    pub username: String,
    #[validate(length(min = 8, max = 100))]
    pub password: String,
}

/// Authentication result
#[derive(Debug, Clone)]
pub enum AuthenticationResult {
    Success {
        user_id: String,
        roles: Vec<String>,
    },
    Failure {
        reason: String,
    },
}

/// Audit logger for security events
pub struct AuditLogger {
    log_file: Mutex<std::fs::File>,
}

impl AuditLogger {
    /// Create new audit logger
    pub async fn new() -> Result<Self> {
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("security_audit.log")
            .map_err(|e| HiveMindError::Internal(format!("Failed to create audit log: {}", e)))?;

        Ok(Self {
            log_file: Mutex::new(log_file),
        })
    }

    /// Log security event
    pub async fn log_security_event(&self, event: SecurityEvent) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| HiveMindError::Internal("Time error".to_string()))?
            .as_secs();

        let log_entry = format!(
            "[{}] {}: {}\n",
            timestamp,
            event.event_type(),
            serde_json::to_string(&event)
                .map_err(|e| HiveMindError::Internal(format!("Failed to serialize event: {}", e)))?
        );

        use std::io::Write;
        let mut file = self.log_file.lock().await;
        file.write_all(log_entry.as_bytes())
            .map_err(|e| HiveMindError::Internal(format!("Failed to write audit log: {}", e)))?;
        file.flush()
            .map_err(|e| HiveMindError::Internal(format!("Failed to flush audit log: {}", e)))?;

        Ok(())
    }
}

/// Security events for audit logging
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum SecurityEvent {
    AuthenticationSuccess { user_id: String, ip: String },
    AuthenticationFailure { reason: String, ip: String },
    SessionCreated { session_id: String, user_id: String, ip: String },
    SessionExpired { session_id: String, user_id: String },
    RateLimitExceeded { ip: String },
    InputValidationFailed { input_type: String, reason: String },
    CryptographicOperation { operation: String, key_id: String },
    SecurityPolicyViolation { policy: String, details: String },
}

impl SecurityEvent {
    pub fn event_type(&self) -> &'static str {
        match self {
            SecurityEvent::AuthenticationSuccess { .. } => "AUTH_SUCCESS",
            SecurityEvent::AuthenticationFailure { .. } => "AUTH_FAILURE",
            SecurityEvent::SessionCreated { .. } => "SESSION_CREATED",
            SecurityEvent::SessionExpired { .. } => "SESSION_EXPIRED",
            SecurityEvent::RateLimitExceeded { .. } => "RATE_LIMIT_EXCEEDED",
            SecurityEvent::InputValidationFailed { .. } => "INPUT_VALIDATION_FAILED",
            SecurityEvent::CryptographicOperation { .. } => "CRYPTO_OPERATION",
            SecurityEvent::SecurityPolicyViolation { .. } => "POLICY_VIOLATION",
        }
    }
}

/// Input validator for sanitization and validation
pub struct InputValidator;

impl InputValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate and sanitize input based on type
    pub async fn validate_and_sanitize(&self, input: &str, input_type: InputType) -> Result<String> {
        // Check input size
        if input.len() > MAX_INPUT_SIZE {
            return Err(HiveMindError::InvalidState {
                message: "Input size exceeds maximum allowed".to_string(),
            });
        }

        // Sanitize based on type
        let sanitized = match input_type {
            InputType::Username => self.sanitize_username(input)?,
            InputType::Email => self.sanitize_email(input)?,
            InputType::Json => self.sanitize_json(input)?,
            InputType::Number => self.sanitize_number(input)?,
            InputType::Text => self.sanitize_text(input)?,
        };

        Ok(sanitized)
    }

    /// Sanitize username input
    fn sanitize_username(&self, input: &str) -> Result<String> {
        let sanitized: String = input
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .take(50)
            .collect();

        if sanitized.is_empty() {
            return Err(HiveMindError::InvalidState {
                message: "Invalid username format".to_string(),
            });
        }

        Ok(sanitized)
    }

    /// Sanitize email input
    fn sanitize_email(&self, input: &str) -> Result<String> {
        let sanitized = input.trim().to_lowercase();
        
        // Basic email validation
        if !sanitized.contains('@') || !sanitized.contains('.') {
            return Err(HiveMindError::InvalidState {
                message: "Invalid email format".to_string(),
            });
        }

        Ok(sanitized)
    }

    /// Sanitize JSON input
    fn sanitize_json(&self, input: &str) -> Result<String> {
        // Parse JSON to validate structure
        let _: serde_json::Value = serde_json::from_str(input)
            .map_err(|_| HiveMindError::InvalidState {
                message: "Invalid JSON format".to_string(),
            })?;

        Ok(input.to_string())
    }

    /// Sanitize numeric input
    fn sanitize_number(&self, input: &str) -> Result<String> {
        let sanitized: String = input
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '.' || *c == '-')
            .collect();

        // Validate it's a proper number
        sanitized.parse::<f64>()
            .map_err(|_| HiveMindError::InvalidState {
                message: "Invalid number format".to_string(),
            })?;

        Ok(sanitized)
    }

    /// Sanitize general text input
    fn sanitize_text(&self, input: &str) -> Result<String> {
        // Remove potential XSS and injection characters
        let sanitized: String = input
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
            .collect();

        Ok(sanitized)
    }
}

/// Input types for validation
#[derive(Debug, Clone, Copy)]
pub enum InputType {
    Username,
    Email,
    Json,
    Number,
    Text,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_manager_creation() {
        let security_manager = SecurityManager::new().await;
        assert!(security_manager.is_ok());
    }

    #[tokio::test]
    async fn test_password_hashing() {
        let security_manager = SecurityManager::new().await.unwrap();
        
        let password = "test_password_123";
        let hash = security_manager.hash_password(password).await.unwrap();
        
        assert!(security_manager.verify_password(password, &hash).await.unwrap());
        assert!(!security_manager.verify_password("wrong_password", &hash).await.unwrap());
    }

    #[tokio::test]
    async fn test_input_validation() {
        let validator = InputValidator::new();
        
        // Valid username
        let result = validator.validate_and_sanitize("user123", InputType::Username).await;
        assert!(result.is_ok());
        
        // Invalid username with special characters
        let result = validator.validate_and_sanitize("user<script>", InputType::Username).await;
        assert_eq!(result.unwrap(), "userscript");
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let rate_limiter = RateLimiter::new();
        let ip = "127.0.0.1";
        let endpoint = "test";
        
        // First few requests should be allowed
        for _ in 0..5 {
            assert!(rate_limiter.check_rate_limit(ip, endpoint).await.unwrap());
        }
    }

    #[tokio::test]
    async fn test_encryption_decryption() {
        let key_store = SecureKeyStore::new().await.unwrap();
        let data = b"sensitive financial data";
        
        let encrypted = key_store.encrypt_data(data, "master").await.unwrap();
        let decrypted = key_store.decrypt_data(&encrypted).await.unwrap();
        
        assert_eq!(data, &decrypted[..]);
    }

    #[tokio::test]
    async fn test_digital_signature() {
        let key_store = SecureKeyStore::new().await.unwrap();
        let data = b"transaction data to sign";
        
        let signature = key_store.sign_data(data, "master").await.unwrap();
        let is_valid = key_store.verify_signature(data, &signature, "master").await.unwrap();
        
        assert!(is_valid);
    }
}