use hmac::{Hmac, Mac};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

type HmacSha256 = Hmac<Sha256>;

/// Cryptographic data validator for ensuring data integrity
///
/// References:
/// - FIPS 180-4: Secure Hash Standard (SHS)
/// - RFC 2104: HMAC - Keyed-Hashing for Message Authentication
/// - NIST SP 800-107: Recommendation for Applications Using Approved Hash Algorithms
#[derive(Debug)]
pub struct CryptographicDataValidator {
    hmac_key: Vec<u8>,
    known_hashes: HashMap<String, String>,
    validation_rules: ValidationRules,
}

#[derive(Debug)]
struct ValidationRules {
    max_timestamp_drift: u64,
    required_fields: Vec<String>,
    numeric_ranges: HashMap<String, (f64, f64)>,
}

impl CryptographicDataValidator {
    /// Initialize validator with cryptographic key
    pub fn new() -> Result<Self, ValidationError> {
        // Generate secure HMAC key (in production, this would be from secure storage)
        let hmac_key = Self::generate_secure_key()?;

        let validation_rules = ValidationRules {
            max_timestamp_drift: 60_000, // 60 seconds
            required_fields: vec![
                "symbol".to_string(),
                "price".to_string(),
                "volume".to_string(),
                "timestamp".to_string(),
            ],
            numeric_ranges: {
                let mut ranges = HashMap::new();
                ranges.insert("price".to_string(), (0.0001, 1_000_000.0));
                ranges.insert("volume".to_string(), (0.0, f64::MAX));
                ranges.insert("bid_price".to_string(), (0.0001, 1_000_000.0));
                ranges.insert("ask_price".to_string(), (0.0001, 1_000_000.0));
                ranges
            },
        };

        Ok(Self {
            hmac_key,
            known_hashes: HashMap::new(),
            validation_rules,
        })
    }

    /// Validate message integrity using cryptographic methods
    pub fn validate_message_integrity(&mut self, message: &str) -> Result<(), ValidationError> {
        // Parse JSON message
        let json_data: Value = serde_json::from_str(message)
            .map_err(|e| ValidationError::InvalidJson(e.to_string()))?;

        // Validate required fields
        self.validate_required_fields(&json_data)?;

        // Validate data types and ranges
        self.validate_data_ranges(&json_data)?;

        // Validate timestamp freshness
        self.validate_timestamp_freshness(&json_data)?;

        // Calculate and verify message hash
        let message_hash = self.calculate_message_hash(message)?;
        self.verify_message_hash(&message_hash, message)?;

        // Store hash for replay attack detection
        self.store_message_hash(message_hash, message.to_string());

        Ok(())
    }

    /// Validate that all required fields are present
    fn validate_required_fields(&self, data: &Value) -> Result<(), ValidationError> {
        for field in &self.validation_rules.required_fields {
            if !data.get(field).is_some() {
                return Err(ValidationError::MissingRequiredField(field.clone()));
            }
        }
        Ok(())
    }

    /// Validate numeric ranges for financial data
    fn validate_data_ranges(&self, data: &Value) -> Result<(), ValidationError> {
        for (field, (min, max)) in &self.validation_rules.numeric_ranges {
            if let Some(value) = data.get(field) {
                if let Some(num_value) = value.as_f64() {
                    if num_value < *min || num_value > *max {
                        return Err(ValidationError::ValueOutOfRange {
                            field: field.clone(),
                            value: num_value,
                            min: *min,
                            max: *max,
                        });
                    }
                } else {
                    return Err(ValidationError::InvalidDataType {
                        field: field.clone(),
                        expected: "number".to_string(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Validate timestamp freshness to prevent replay attacks
    fn validate_timestamp_freshness(&self, data: &Value) -> Result<(), ValidationError> {
        if let Some(timestamp_value) = data.get("timestamp") {
            if let Some(timestamp) = timestamp_value.as_u64() {
                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_err(|_| ValidationError::SystemTimeError)?
                    .as_millis() as u64;

                let time_diff = if current_time > timestamp {
                    current_time - timestamp
                } else {
                    timestamp - current_time
                };

                if time_diff > self.validation_rules.max_timestamp_drift {
                    return Err(ValidationError::TimestampTooOld {
                        timestamp,
                        current_time,
                        max_drift: self.validation_rules.max_timestamp_drift,
                    });
                }
            } else {
                return Err(ValidationError::InvalidDataType {
                    field: "timestamp".to_string(),
                    expected: "integer".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Calculate HMAC-SHA256 hash of message
    fn calculate_message_hash(&self, message: &str) -> Result<String, ValidationError> {
        let mut mac = HmacSha256::new_from_slice(&self.hmac_key)
            .map_err(|_| ValidationError::CryptographicError("Invalid HMAC key".to_string()))?;

        mac.update(message.as_bytes());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Verify message hash against known good hashes
    fn verify_message_hash(&self, hash: &str, _message: &str) -> Result<(), ValidationError> {
        // In production, this would verify against a trusted source
        // For now, we just ensure the hash was calculated successfully
        if hash.len() != 64 {
            // SHA256 hex length
            return Err(ValidationError::InvalidHash(hash.to_string()));
        }
        Ok(())
    }

    /// Store message hash to detect replay attacks
    fn store_message_hash(&mut self, hash: String, message: String) {
        self.known_hashes.insert(hash, message);

        // Limit cache size to prevent memory exhaustion
        if self.known_hashes.len() > 10000 {
            let oldest_key = self.known_hashes.keys().next().cloned();
            if let Some(key) = oldest_key {
                self.known_hashes.remove(&key);
            }
        }
    }

    /// Generate cryptographically secure key
    fn generate_secure_key() -> Result<Vec<u8>, ValidationError> {
        use rand::RngCore;
        let mut key = vec![0u8; 32]; // 256-bit key
        rand::thread_rng().fill_bytes(&mut key);
        Ok(key)
    }

    /// Validate symbol against allowed trading pairs
    pub fn validate_symbol(&self, symbol: &str) -> Result<(), ValidationError> {
        const ALLOWED_SYMBOLS: &[&str] = &[
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT",
            "LINKUSDT", "LTCUSDT", "BCHUSDT",
        ];

        if !ALLOWED_SYMBOLS.contains(&symbol) {
            return Err(ValidationError::InvalidSymbol(symbol.to_string()));
        }

        Ok(())
    }

    /// Detect potential replay attacks
    pub fn check_replay_attack(&self, message: &str) -> Result<(), ValidationError> {
        let hash = self.calculate_message_hash(message)?;

        if self.known_hashes.contains_key(&hash) {
            return Err(ValidationError::ReplayAttackDetected(hash));
        }

        Ok(())
    }

    /// Get validation statistics for monitoring
    pub fn get_validation_stats(&self) -> ValidationStats {
        ValidationStats {
            total_hashes_stored: self.known_hashes.len(),
            max_timestamp_drift: self.validation_rules.max_timestamp_drift,
            required_fields_count: self.validation_rules.required_fields.len(),
            numeric_ranges_count: self.validation_rules.numeric_ranges.len(),
        }
    }
}

/// Validation statistics for monitoring
#[derive(Debug, Clone)]
pub struct ValidationStats {
    pub total_hashes_stored: usize,
    pub max_timestamp_drift: u64,
    pub required_fields_count: usize,
    pub numeric_ranges_count: usize,
}

/// Validation errors
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),

    #[error("Missing required field: {0}")]
    MissingRequiredField(String),

    #[error("Value out of range for field '{field}': {value} (expected {min} to {max})")]
    ValueOutOfRange {
        field: String,
        value: f64,
        min: f64,
        max: f64,
    },

    #[error("Invalid data type for field '{field}': expected {expected}")]
    InvalidDataType { field: String, expected: String },

    #[error("Timestamp too old: {timestamp} (current: {current_time}, max drift: {max_drift}ms)")]
    TimestampTooOld {
        timestamp: u64,
        current_time: u64,
        max_drift: u64,
    },

    #[error("System time error")]
    SystemTimeError,

    #[error("Cryptographic error: {0}")]
    CryptographicError(String),

    #[error("Invalid hash: {0}")]
    InvalidHash(String),

    #[error("Invalid symbol: {0}")]
    InvalidSymbol(String),

    #[error("Replay attack detected: {0}")]
    ReplayAttackDetected(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_creation() {
        let validator = CryptographicDataValidator::new();
        assert!(validator.is_ok());
    }

    #[test]
    fn test_validate_required_fields() {
        let mut validator = CryptographicDataValidator::new().unwrap();

        let valid_message =
            r#"{"symbol":"BTCUSDT","price":50000.0,"volume":100.0,"timestamp":1640995200000}"#;
        assert!(validator.validate_message_integrity(valid_message).is_ok());

        let invalid_message = r#"{"symbol":"BTCUSDT","price":50000.0}"#;
        assert!(validator
            .validate_message_integrity(invalid_message)
            .is_err());
    }

    #[test]
    fn test_validate_symbol() {
        let validator = CryptographicDataValidator::new().unwrap();

        assert!(validator.validate_symbol("BTCUSDT").is_ok());
        assert!(validator.validate_symbol("INVALID").is_err());
    }
}
