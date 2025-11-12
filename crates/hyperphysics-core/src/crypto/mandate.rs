//! # Payment Mandate Structure
//!
//! Defines the payment authorization structure that agents sign and verify.
//! Supports spend caps, time windows, and merchant restrictions.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Payment mandate authorizing agent spending
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentMandate {
    /// Agent identifier requesting authorization
    pub agent_id: String,

    /// Holder/supervisor identifier (e.g., Queen Seraphina)
    pub holder_id: String,

    /// Maximum amount in minor currency units (e.g., cents)
    pub amount_cents: u64,

    /// Currency code (ISO 4217)
    pub currency: String,

    /// Spend period for the cap
    pub period: Period,

    /// Type of mandate authorization
    pub kind: MandateKind,

    /// Expiration timestamp (UTC)
    pub expires_at: DateTime<Utc>,

    /// Allowed merchant hostnames
    pub merchant_allow: Vec<String>,

    /// Blocked merchant hostnames (optional)
    #[serde(default)]
    pub merchant_block: Vec<String>,
}

/// Spend period for mandate caps
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Period {
    /// Single-use authorization
    Single,

    /// Daily spending limit
    Daily,

    /// Weekly spending limit
    Weekly,

    /// Monthly spending limit
    Monthly,
}

/// Type of payment mandate
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MandateKind {
    /// Intent-based authorization (high-level purchase intent)
    Intent,

    /// Cart-based authorization (specific line items)
    Cart,
}

impl PaymentMandate {
    /// Create new payment mandate
    ///
    /// # Arguments
    /// * `agent_id` - Agent requesting authorization
    /// * `holder_id` - Supervisor granting authorization
    /// * `amount_cents` - Maximum amount in minor units
    /// * `currency` - Currency code (e.g., "USD")
    /// * `period` - Spending period
    /// * `kind` - Mandate type
    /// * `expires_at` - Expiration time
    /// * `merchant_allow` - Allowed merchants
    pub fn new(
        agent_id: String,
        holder_id: String,
        amount_cents: u64,
        currency: String,
        period: Period,
        kind: MandateKind,
        expires_at: DateTime<Utc>,
        merchant_allow: Vec<String>,
    ) -> Self {
        Self {
            agent_id,
            holder_id,
            amount_cents,
            currency,
            period,
            kind,
            expires_at,
            merchant_allow,
            merchant_block: Vec::new(),
        }
    }

    /// Add blocked merchant to mandate
    pub fn block_merchant(&mut self, merchant: String) {
        self.merchant_block.push(merchant);
    }

    /// Check if merchant is allowed
    pub fn is_merchant_allowed(&self, merchant: &str) -> bool {
        if self.merchant_block.contains(&merchant.to_string()) {
            return false;
        }

        if self.merchant_allow.is_empty() {
            return true;  // No restrictions
        }

        self.merchant_allow.contains(&merchant.to_string())
    }

    /// Check if mandate is expired
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }

    /// Serialize mandate to bytes for signing
    ///
    /// Returns canonical JSON representation
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self)
            .expect("PaymentMandate serialization should never fail")
    }

    /// Deserialize mandate from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes)
            .map_err(|e| format!("Failed to deserialize mandate: {}", e))
    }

    /// Get human-readable amount
    pub fn amount_major(&self) -> f64 {
        self.amount_cents as f64 / 100.0
    }
}

/// Signed payment mandate with cryptographic proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedMandate {
    /// The mandate data
    pub mandate: PaymentMandate,

    /// Hex-encoded Ed25519 signature
    pub signature: String,

    /// Hex-encoded public key of signer
    pub signer_public_key: String,
}

impl SignedMandate {
    /// Create new signed mandate
    pub fn new(
        mandate: PaymentMandate,
        signature: String,
        signer_public_key: String,
    ) -> Self {
        Self {
            mandate,
            signature,
            signer_public_key,
        }
    }

    /// Verify signature on mandate
    pub fn verify(&self) -> Result<(), String> {
        use ed25519_dalek::{VerifyingKey, Signature, Verifier};

        let public_bytes = hex::decode(&self.signer_public_key)
            .map_err(|e| format!("Invalid public key hex: {}", e))?;

        let public_key_array: [u8; 32] = public_bytes
            .try_into()
            .map_err(|_| "Public key must be exactly 32 bytes".to_string())?;

        let verifying_key = VerifyingKey::from_bytes(&public_key_array)
            .map_err(|e| format!("Invalid public key: {}", e))?;

        let sig_bytes = hex::decode(&self.signature)
            .map_err(|e| format!("Invalid signature hex: {}", e))?;

        let sig_array: [u8; 64] = sig_bytes
            .try_into()
            .map_err(|_| "Signature must be exactly 64 bytes".to_string())?;

        let signature = Signature::from_bytes(&sig_array);

        let message = self.mandate.to_bytes();

        verifying_key.verify(&message, &signature)
            .map_err(|e| format!("Signature verification failed: {}", e))
    }

    /// Check all mandate guards (expiration, merchant rules)
    pub fn check_guards(&self, merchant: &str) -> Result<(), String> {
        if self.mandate.is_expired() {
            return Err("Mandate has expired".to_string());
        }

        if !self.mandate.is_merchant_allowed(merchant) {
            return Err(format!("Merchant '{}' is not allowed", merchant));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn test_mandate_creation() {
        let mandate = PaymentMandate::new(
            "agent-001".to_string(),
            "queen-seraphina".to_string(),
            10000,  // $100.00
            "USD".to_string(),
            Period::Daily,
            MandateKind::Intent,
            Utc::now() + Duration::hours(24),
            vec!["example.com".to_string()],
        );

        assert_eq!(mandate.amount_major(), 100.0);
        assert_eq!(mandate.period, Period::Daily);
        assert!(!mandate.is_expired());
    }

    #[test]
    fn test_merchant_filtering() {
        let mut mandate = PaymentMandate::new(
            "agent-001".to_string(),
            "queen".to_string(),
            5000,
            "USD".to_string(),
            Period::Single,
            MandateKind::Cart,
            Utc::now() + Duration::hours(1),
            vec!["allowed.com".to_string()],
        );

        assert!(mandate.is_merchant_allowed("allowed.com"));
        assert!(!mandate.is_merchant_allowed("forbidden.com"));

        mandate.block_merchant("blocked.com".to_string());
        assert!(!mandate.is_merchant_allowed("blocked.com"));
    }

    #[test]
    fn test_serialization() {
        let mandate = PaymentMandate::new(
            "agent-001".to_string(),
            "queen".to_string(),
            1000,
            "USD".to_string(),
            Period::Weekly,
            MandateKind::Intent,
            Utc::now() + Duration::days(7),
            vec![],
        );

        let bytes = mandate.to_bytes();
        let restored = PaymentMandate::from_bytes(&bytes).unwrap();

        assert_eq!(mandate.agent_id, restored.agent_id);
        assert_eq!(mandate.amount_cents, restored.amount_cents);
    }
}
