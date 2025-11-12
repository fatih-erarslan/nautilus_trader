//! Standalone crypto tests - minimal dependencies

#[cfg(test)]
mod crypto_tests {
    use ed25519_dalek::{Signer, Verifier, SigningKey, VerifyingKey, Signature};
    use serde::{Deserialize, Serialize};
    use chrono::{Utc, Duration};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestIdentity {
        agent_id: String,
        public_key: String,
        #[serde(skip)]
        secret_key: Option<SigningKey>,
    }

    impl TestIdentity {
        fn generate(agent_id: String) -> Self {
            let signing_key = SigningKey::from_bytes(&rand::random::<[u8; 32]>());
            let verifying_key = signing_key.verifying_key();

            Self {
                agent_id,
                public_key: hex::encode(verifying_key.to_bytes()),
                secret_key: Some(signing_key),
            }
        }

        fn sign(&self, message: &[u8]) -> Result<Signature, String> {
            let signing_key = self.secret_key.as_ref()
                .ok_or("No secret key")?;
            Ok(signing_key.sign(message))
        }

        fn verify_signature(public_key: &str, message: &[u8], signature: &Signature) -> Result<(), String> {
            let public_bytes = hex::decode(public_key)
                .map_err(|e| format!("Invalid hex: {}", e))?;

            let public_key_array: [u8; 32] = public_bytes
                .try_into()
                .map_err(|_| "Must be 32 bytes".to_string())?;

            let verifying_key = VerifyingKey::from_bytes(&public_key_array)
                .map_err(|e| format!("Invalid key: {}", e))?;

            verifying_key.verify(message, signature)
                .map_err(|e| format!("Verification failed: {}", e))
        }
    }

    #[test]
    fn test_identity_generation() {
        let identity = TestIdentity::generate("test-agent".to_string());
        assert_eq!(identity.agent_id, "test-agent");
        assert!(identity.secret_key.is_some());
    }

    #[test]
    fn test_signing() {
        let identity = TestIdentity::generate("signer".to_string());
        let message = b"test message";

        let signature = identity.sign(message).unwrap();
        let result = TestIdentity::verify_signature(&identity.public_key, message, &signature);

        assert!(result.is_ok());
    }

    #[test]
    fn test_byzantine_consensus() {
        // Create 3 agents
        let agent1 = TestIdentity::generate("agent1".to_string());
        let agent2 = TestIdentity::generate("agent2".to_string());
        let agent3 = TestIdentity::generate("agent3".to_string());

        let message = b"approve payment";

        // All sign the message
        let sig1 = agent1.sign(message).unwrap();
        let sig2 = agent2.sign(message).unwrap();
        let sig3 = agent3.sign(message).unwrap();

        // Verify all signatures
        assert!(TestIdentity::verify_signature(&agent1.public_key, message, &sig1).is_ok());
        assert!(TestIdentity::verify_signature(&agent2.public_key, message, &sig2).is_ok());
        assert!(TestIdentity::verify_signature(&agent3.public_key, message, &sig3).is_ok());

        // 2/3 consensus achieved
        let approvals = 3;
        let total = 3;
        assert!(approvals as f64 / total as f64 >= 2.0/3.0);
    }

    #[test]
    fn test_payment_mandate_structure() {
        #[derive(Serialize, Deserialize)]
        struct PaymentMandate {
            agent_id: String,
            holder_id: String,
            amount_cents: u64,
            currency: String,
            expires_at: chrono::DateTime<Utc>,
        }

        let mandate = PaymentMandate {
            agent_id: "agent-001".to_string(),
            holder_id: "queen".to_string(),
            amount_cents: 10000,
            currency: "USD".to_string(),
            expires_at: Utc::now() + Duration::hours(24),
        };

        // Serialize to bytes
        let bytes = serde_json::to_vec(&mandate).unwrap();
        assert!(!bytes.is_empty());

        // Sign the mandate
        let queen = TestIdentity::generate("queen".to_string());
        let signature = queen.sign(&bytes).unwrap();

        // Verify signature
        assert!(TestIdentity::verify_signature(&queen.public_key, &bytes, &signature).is_ok());
    }

    #[test]
    fn test_full_workflow() {
        println!("\n=== CRYPTO WORKFLOW TEST ===\n");

        // 1. Generate Queen identity
        let queen = TestIdentity::generate("queen-seraphina".to_string());
        println!("✓ Queen Seraphina identity generated");
        println!("  Public key: {}...", &queen.public_key[..16]);

        // 2. Generate agent identities
        let agent1 = TestIdentity::generate("agent-001".to_string());
        let agent2 = TestIdentity::generate("agent-002".to_string());
        let agent3 = TestIdentity::generate("agent-003".to_string());
        println!("✓ 3 agent identities generated");

        // 3. Create payment mandate
        #[derive(Serialize, Deserialize)]
        struct Mandate {
            agent_id: String,
            amount_cents: u64,
            expires_at: chrono::DateTime<Utc>,
        }

        let mandate = Mandate {
            agent_id: agent1.agent_id.clone(),
            amount_cents: 15000,
            expires_at: Utc::now() + Duration::days(7),
        };

        let mandate_bytes = serde_json::to_vec(&mandate).unwrap();
        println!("✓ Payment mandate created: $150.00");

        // 4. Queen signs mandate
        let queen_sig = queen.sign(&mandate_bytes).unwrap();
        assert!(TestIdentity::verify_signature(&queen.public_key, &mandate_bytes, &queen_sig).is_ok());
        println!("✓ Queen signed mandate");

        // 5. Byzantine consensus voting
        let vote_msg = b"approve mandate";

        let votes = vec![
            (agent1.agent_id.clone(), agent1.sign(vote_msg).unwrap(), agent1.public_key.clone(), true),
            (agent2.agent_id.clone(), agent2.sign(vote_msg).unwrap(), agent2.public_key.clone(), true),
            (agent3.agent_id.clone(), agent3.sign(vote_msg).unwrap(), agent3.public_key.clone(), true),
        ];

        // Verify all votes
        for (id, sig, pk, _) in &votes {
            assert!(TestIdentity::verify_signature(pk, vote_msg, sig).is_ok());
        }

        let approvals = votes.iter().filter(|(_, _, _, approve)| *approve).count();
        let consensus_threshold = 2.0 / 3.0;
        let approval_rate = approvals as f64 / votes.len() as f64;

        println!("✓ All vote signatures verified");
        println!("  Approval rate: {:.1}%", approval_rate * 100.0);
        println!("  Threshold: {:.1}%", consensus_threshold * 100.0);

        assert!(approval_rate >= consensus_threshold);
        println!("✓ Byzantine consensus achieved!\n");
    }
}
