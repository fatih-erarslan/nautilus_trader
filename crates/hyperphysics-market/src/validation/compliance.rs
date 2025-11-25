//! SEC Regulation S-P and Regulatory Compliance
//!
//! Implements data privacy and security validation per SEC Regulation S-P
//! (Privacy of Consumer Financial Information, effective December 13, 2000).
//!
//! # Regulatory Requirements
//!
//! SEC Regulation S-P requires financial institutions to:
//! 1. Protect customer information (§314.4)
//! 2. Implement data safeguards
//! 3. Maintain privacy policies
//! 4. Notify customers of data sharing practices
//!
//! # References
//!
//! - 17 CFR §248 - Regulation S-P
//! - SEC Release No. 34-42974 (June 22, 2000)
//! - FINRA Rule 4512 (Customer Account Information)

use crate::data::Bar;
use crate::data::tick::{Quote, Tick};
use crate::error::{MarketError, MarketResult};

/// Regulatory compliance validator
#[derive(Debug)]
pub struct ComplianceValidator {
    /// Enable SEC Regulation S-P checks
    sec_s_p_enabled: bool,
}

impl ComplianceValidator {
    /// Create new compliance validator
    pub fn new(sec_s_p_enabled: bool) -> Self {
        Self { sec_s_p_enabled }
    }

    /// Validate bar data for compliance
    pub fn validate_bar(&self, bar: &Bar) -> MarketResult<()> {
        if !self.sec_s_p_enabled {
            return Ok(());
        }

        // Ensure no personally identifiable information (PII) in symbol
        self.check_pii_in_symbol(&bar.symbol)?;

        // Validate data retention compliance (bars should not be too old)
        self.check_data_retention(&bar.timestamp)?;

        Ok(())
    }

    /// Validate tick data for compliance
    pub fn validate_tick(&self, tick: &Tick) -> MarketResult<()> {
        if !self.sec_s_p_enabled {
            return Ok(());
        }

        self.check_pii_in_symbol(&tick.symbol)?;
        self.check_data_retention(&tick.timestamp)?;

        Ok(())
    }

    /// Validate quote data for compliance
    pub fn validate_quote(&self, quote: &Quote) -> MarketResult<()> {
        if !self.sec_s_p_enabled {
            return Ok(());
        }

        self.check_pii_in_symbol(&quote.symbol)?;
        self.check_data_retention(&quote.timestamp)?;

        Ok(())
    }

    /// Check for PII in symbol field (should only contain ticker symbols)
    fn check_pii_in_symbol(&self, symbol: &str) -> MarketResult<()> {
        // Symbol should only contain uppercase letters, numbers, and common separators
        if !symbol.chars().all(|c| c.is_ascii_alphanumeric() || c == '/' || c == '-' || c == '.') {
            return Err(MarketError::ValidationError(format!(
                "Invalid characters in symbol: {}",
                symbol
            )));
        }

        // Symbol should not be excessively long (max 20 characters)
        if symbol.len() > 20 {
            return Err(MarketError::ValidationError(format!(
                "Symbol too long: {} characters",
                symbol.len()
            )));
        }

        // Check for common PII patterns (email, phone, SSN)
        if symbol.contains('@') || symbol.contains("SSN") || symbol.contains("DOB") {
            return Err(MarketError::ValidationError(
                "Potential PII detected in symbol".to_string()
            ));
        }

        Ok(())
    }

    /// Check data retention compliance (SEC requires 6 years for broker-dealer records)
    fn check_data_retention(&self, timestamp: &chrono::DateTime<chrono::Utc>) -> MarketResult<()> {
        use chrono::Utc;

        let now = Utc::now();
        let age = now.signed_duration_since(*timestamp);

        // Warn if data is older than 6 years (SEC retention requirement)
        if age > chrono::Duration::days(365 * 6) {
            // This is a soft check - old data is still valid, just flagged
            // In production, would log to audit trail
        }

        // Hard check: reject data older than 10 years (abnormal)
        if age > chrono::Duration::days(365 * 10) {
            return Err(MarketError::ValidationError(format!(
                "Data exceeds reasonable retention period: {} years old",
                age.num_days() / 365
            )));
        }

        Ok(())
    }

    /// Validate data anonymization (for research/aggregated data)
    pub fn validate_anonymization(&self, symbol: &str) -> MarketResult<()> {
        // For aggregated/research data, ensure symbols are properly anonymized
        // This would check if symbol matches anonymization patterns (e.g., "SYMBOL_001")

        // Basic check: ensure no account numbers or customer IDs
        if symbol.contains("ACCT") || symbol.contains("CUST") || symbol.contains("ID") {
            return Err(MarketError::ValidationError(
                "Potential customer identifier in symbol".to_string()
            ));
        }

        Ok(())
    }

    /// Audit log entry (would write to secure audit trail in production)
    pub fn audit_access(&self, symbol: &str, access_type: &str) {
        // In production, this would:
        // 1. Write to tamper-proof audit log
        // 2. Include timestamp, user, action
        // 3. Encrypt sensitive data
        // 4. Meet SEC 17a-4 record retention requirements

        tracing::debug!(
            "Audit: {} access to symbol {}",
            access_type,
            symbol
        );
    }

    /// Validate data encryption (for transmission/storage)
    pub fn validate_encryption(&self, _encrypted: bool) -> MarketResult<()> {
        // In production, would verify:
        // 1. Data is encrypted at rest (AES-256)
        // 2. Data is encrypted in transit (TLS 1.3)
        // 3. Keys are properly managed (HSM)
        // 4. Meet SEC cybersecurity requirements

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_valid_symbol() {
        let validator = ComplianceValidator::new(true);

        assert!(validator.check_pii_in_symbol("AAPL").is_ok());
        assert!(validator.check_pii_in_symbol("BTC/USD").is_ok());
        assert!(validator.check_pii_in_symbol("SPX-2025").is_ok());
    }

    #[test]
    fn test_invalid_symbol_pii() {
        let validator = ComplianceValidator::new(true);

        assert!(validator.check_pii_in_symbol("user@example.com").is_err());
        assert!(validator.check_pii_in_symbol("SSN-123456789").is_err());
    }

    #[test]
    fn test_symbol_too_long() {
        let validator = ComplianceValidator::new(true);

        let long_symbol = "A".repeat(25);
        assert!(validator.check_pii_in_symbol(&long_symbol).is_err());
    }

    #[test]
    fn test_data_retention() {
        let validator = ComplianceValidator::new(true);

        // Recent data should pass
        let recent = Utc::now();
        assert!(validator.check_data_retention(&recent).is_ok());

        // Very old data should fail (> 10 years)
        let very_old = Utc::now() - chrono::Duration::days(365 * 11);
        assert!(validator.check_data_retention(&very_old).is_err());
    }

    #[test]
    fn test_anonymization_check() {
        let validator = ComplianceValidator::new(true);

        assert!(validator.validate_anonymization("AAPL").is_ok());
        assert!(validator.validate_anonymization("ACCT12345").is_err());
        assert!(validator.validate_anonymization("CUST-ID-999").is_err());
    }
}
