//! Standalone crypto module integration tests

use hyperphysics_core::crypto::{
    identity::AgentIdentity,
    mandate::{PaymentMandate, Period, MandateKind, SignedMandate},
    consensus::{ByzantineConsensus, Vote},
};
use chrono::{Utc, Duration};

#[test]
fn test_agent_identity_generation() {
    let identity = AgentIdentity::generate("test-agent".to_string());
    assert_eq!(identity.agent_id(), "test-agent");
    assert!(identity.can_sign());
}

#[test]
fn test_sign_and_verify() {
    let identity = AgentIdentity::generate("signer".to_string());
    let message = b"payment mandate data";

    let signature = identity.sign(message).unwrap();
    let result = AgentIdentity::verify(&identity.export_public_key(), message, &signature);

    assert!(result.is_ok());
}

#[test]
fn test_payment_mandate_creation() {
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
    assert!(!mandate.is_expired());
}

#[test]
fn test_signed_mandate() {
    let queen = AgentIdentity::generate("queen-seraphina".to_string());

    let mandate = PaymentMandate::new(
        "agent-001".to_string(),
        queen.agent_id().to_string(),
        5000,
        "USD".to_string(),
        Period::Single,
        MandateKind::Cart,
        Utc::now() + Duration::hours(1),
        vec!["trusted-merchant.com".to_string()],
    );

    let mandate_bytes = mandate.to_bytes();
    let signature = queen.sign(&mandate_bytes).unwrap();

    let signed = SignedMandate::new(
        mandate,
        hex::encode(signature.to_bytes()),
        queen.export_public_key(),
    );

    assert!(signed.verify().is_ok());
    assert!(signed.check_guards("trusted-merchant.com").is_ok());
}

#[test]
fn test_byzantine_consensus() {
    let consensus = ByzantineConsensus::new(2.0/3.0);

    let agent1 = AgentIdentity::generate("agent1".to_string());
    let agent2 = AgentIdentity::generate("agent2".to_string());
    let agent3 = AgentIdentity::generate("agent3".to_string());

    let message = b"approve payment mandate";

    let vote1 = Vote {
        agent_id: agent1.agent_id().to_string(),
        approve: true,
        signature: hex::encode(agent1.sign(message).unwrap().to_bytes()),
        public_key: agent1.export_public_key(),
    };

    let vote2 = Vote {
        agent_id: agent2.agent_id().to_string(),
        approve: true,
        signature: hex::encode(agent2.sign(message).unwrap().to_bytes()),
        public_key: agent2.export_public_key(),
    };

    let vote3 = Vote {
        agent_id: agent3.agent_id().to_string(),
        approve: false,
        signature: hex::encode(agent3.sign(message).unwrap().to_bytes()),
        public_key: agent3.export_public_key(),
    };

    let votes = vec![vote1, vote2, vote3];

    // Verify all signatures
    let verify_result = consensus.verify_votes(&votes, message);
    assert!(verify_result.is_ok());

    // Check consensus (2/3 approval threshold met)
    let result = consensus.check_consensus(&votes);
    assert!(result.approved);
    assert_eq!(result.votes_for, 2);
    assert_eq!(result.votes_against, 1);
}

#[test]
fn test_full_payment_workflow() {
    // 1. Generate identities
    let queen = AgentIdentity::generate("queen-seraphina".to_string());
    let agent1 = AgentIdentity::generate("agent-001".to_string());
    let agent2 = AgentIdentity::generate("agent-002".to_string());
    let agent3 = AgentIdentity::generate("agent-003".to_string());

    // 2. Create payment mandate
    let mandate = PaymentMandate::new(
        agent1.agent_id().to_string(),
        queen.agent_id().to_string(),
        15000,  // $150.00
        "USD".to_string(),
        Period::Weekly,
        MandateKind::Intent,
        Utc::now() + Duration::days(7),
        vec!["authorized-vendor.com".to_string()],
    );

    // 3. Queen signs the mandate
    let mandate_bytes = mandate.to_bytes();
    let queen_signature = queen.sign(&mandate_bytes).unwrap();

    let signed_mandate = SignedMandate::new(
        mandate.clone(),
        hex::encode(queen_signature.to_bytes()),
        queen.export_public_key(),
    );

    // 4. Verify queen's signature
    assert!(signed_mandate.verify().is_ok());

    // 5. Byzantine consensus voting
    let consensus = ByzantineConsensus::new(0.67);
    let vote_message = format!("approve mandate for {}", mandate.agent_id).into_bytes();

    let votes = vec![
        Vote {
            agent_id: agent1.agent_id().to_string(),
            approve: true,
            signature: hex::encode(agent1.sign(&vote_message).unwrap().to_bytes()),
            public_key: agent1.export_public_key(),
        },
        Vote {
            agent_id: agent2.agent_id().to_string(),
            approve: true,
            signature: hex::encode(agent2.sign(&vote_message).unwrap().to_bytes()),
            public_key: agent2.export_public_key(),
        },
        Vote {
            agent_id: agent3.agent_id().to_string(),
            approve: true,
            signature: hex::encode(agent3.sign(&vote_message).unwrap().to_bytes()),
            public_key: agent3.export_public_key(),
        },
    ];

    // Verify all vote signatures
    assert!(consensus.verify_votes(&votes, &vote_message).is_ok());

    // Check consensus
    let result = consensus.check_consensus(&votes);
    assert!(result.approved);
    assert_eq!(result.votes_for, 3);
    assert_eq!(result.approval_rate, 1.0);
}
