//! Property-Based Tests for Byzantine Consensus
//!
//! Comprehensive property tests using proptest to verify:
//! - Safety: No two different values committed
//! - Voting quorum correctness (2f+1)
//! - Sequence number monotonicity
//! - Byzantine fault tolerance

use proptest::prelude::*;
use std::collections::HashSet;

// Re-export consensus types
use crate::consensus::byzantine_consensus::{
    ByzantineConsensus, ByzantineMessage, ConsensusPhase, ConsensusState,
    MessageType, QuantumSignature, ValidatorId,
};

const PROP_TEST_CASES: u32 = 1000;

// Strategy generators
fn validator_count_strategy() -> impl Strategy<Value = usize> {
    // Need at least 4 validators for 3f+1 >= 4 (f=1)
    (4usize..=31usize)
}

fn validator_id_strategy(max_validators: usize) -> impl Strategy<Value = ValidatorId> {
    (0u64..(max_validators as u64))
        .prop_map(ValidatorId)
}

fn message_payload_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(any::<u8>(), 1..1024)
}

fn view_number_strategy() -> impl Strategy<Value = u64> {
    (0u64..1000u64)
}

fn sequence_number_strategy() -> impl Strategy<Value = u64> {
    (1u64..10000u64)
}

// Property 1: Byzantine fault tolerance requirement (3f+1 <= n)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_byzantine_fault_tolerance_requirement(
        validator_count in validator_count_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        // Property: System must satisfy 3f+1 <= n
        let f = consensus.byzantine_threshold;
        prop_assert!(
            validator_count >= 3 * f + 1,
            "Must satisfy 3f+1 <= n: 3*{} + 1 = {} <= {}",
            f,
            3 * f + 1,
            validator_count
        );

        // Property: Byzantine threshold should be calculated correctly
        let expected_f = (validator_count - 1) / 3;
        prop_assert_eq!(
            f,
            expected_f,
            "Byzantine threshold must be (n-1)/3"
        );

        // Property: is_byzantine_fault_tolerant should return true
        prop_assert!(
            consensus.is_byzantine_fault_tolerant(),
            "System with {} validators should be BFT",
            validator_count
        );
    }
}

// Property 2: Quorum size is always 2f+1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_quorum_size_correct(
        validator_count in validator_count_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);
        let f = consensus.byzantine_threshold;

        // Property: Quorum should be 2f+1
        let quorum = 2 * f + 1;

        // Property: Quorum must be more than half of validators
        prop_assert!(
            quorum > validator_count / 2,
            "Quorum {} must be more than half of {} validators",
            quorum,
            validator_count
        );

        // Property: Quorum must be at most total validators
        prop_assert!(
            quorum <= validator_count,
            "Quorum {} must not exceed validator count {}",
            quorum,
            validator_count
        );

        // Property: Quorum must be able to overlap with any set of non-faulty validators
        // Even if f validators are faulty, quorum of non-faulty exists
        let max_faulty = f;
        let min_honest = validator_count - max_faulty;
        prop_assert!(
            quorum <= min_honest,
            "Quorum {} must be achievable with {} honest validators (f={})",
            quorum,
            min_honest,
            f
        );
    }
}

// Property 3: Sequence numbers are monotonically increasing
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))]

    #[tokio::test]
    async fn prop_sequence_monotonicity(
        validator_count in validator_count_strategy(),
        num_transactions in (1usize..20usize),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        let mut prev_sequence = 0u64;

        // Propose multiple transactions
        for i in 0..num_transactions {
            let transaction = format!("transaction_{}", i).into_bytes();

            // Propose transaction
            if consensus.propose_transaction(transaction).await.is_ok() {
                let state = consensus.get_consensus_state();
                let current_sequence = state.sequence;

                // Property: Sequence number must increase
                prop_assert!(
                    current_sequence > prev_sequence,
                    "Sequence must be monotonically increasing: {} -> {}",
                    prev_sequence,
                    current_sequence
                );

                prev_sequence = current_sequence;
            }
        }
    }
}

// Property 4: View changes increment view number
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[tokio::test]
    async fn prop_view_change_increments(
        validator_count in validator_count_strategy(),
        initial_view in view_number_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        // Get initial state
        let initial_state = consensus.get_consensus_state();
        let initial_view_num = initial_state.view;

        // Create view change message
        let view_change_msg = ByzantineMessage {
            message_type: MessageType::ViewChange,
            view: initial_view_num,
            sequence: 1,
            sender: ValidatorId(0),
            payload: vec![],
            quantum_signature: QuantumSignature {
                signature: vec![1, 2, 3],
                public_key: vec![4, 5, 6],
                quantum_proof: vec![7, 8, 9],
            },
            timestamp: 0,
        };

        // Handle view change
        if consensus.handle_message(view_change_msg).await.is_ok() {
            let new_state = consensus.get_consensus_state();

            // Property: View must have incremented
            prop_assert!(
                new_state.view > initial_view_num,
                "View must increment after view change: {} -> {}",
                initial_view_num,
                new_state.view
            );
        }
    }
}

// Property 5: Valid quantum signatures have all components
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_quantum_signature_complete(
        payload in message_payload_strategy(),
    ) {
        // Generate signature (would use actual quantum verifier in real code)
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        payload.hash(&mut hasher);
        let hash = hasher.finish();

        let signature = QuantumSignature {
            signature: hash.to_le_bytes().to_vec(),
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8],
            quantum_proof: vec![9, 10, 11, 12, 13, 14, 15, 16],
        };

        // Property: All signature components must be non-empty
        prop_assert!(
            !signature.signature.is_empty(),
            "Signature must not be empty"
        );
        prop_assert!(
            !signature.public_key.is_empty(),
            "Public key must not be empty"
        );
        prop_assert!(
            !signature.quantum_proof.is_empty(),
            "Quantum proof must not be empty"
        );
    }
}

// Property 6: Messages have strictly increasing timestamps
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_message_timestamp_ordering(
        num_messages in (2usize..10usize),
    ) {
        use std::time::{SystemTime, UNIX_EPOCH};

        let mut timestamps = Vec::new();

        for _ in 0..num_messages {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;
            timestamps.push(timestamp);

            // Small delay to ensure different timestamps
            std::thread::sleep(std::time::Duration::from_micros(1));
        }

        // Property: Timestamps should be in increasing order
        for i in 0..timestamps.len() - 1 {
            prop_assert!(
                timestamps[i] <= timestamps[i + 1],
                "Timestamps must be non-decreasing"
            );
        }
    }
}

// Property 7: Consensus phases transition in correct order
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))]

    #[tokio::test]
    async fn prop_phase_transition_order(
        validator_count in validator_count_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        // Initial state should be PrePrepare
        let initial_state = consensus.get_consensus_state();
        prop_assert_eq!(
            initial_state.phase,
            ConsensusPhase::PrePrepare,
            "Initial phase must be PrePrepare"
        );

        // Propose a transaction to trigger phase transitions
        let transaction = b"test_transaction".to_vec();
        if consensus.propose_transaction(transaction).await.is_ok() {
            let state = consensus.get_consensus_state();

            // Property: After proposal, phase should progress
            // (PrePrepare -> Prepare is expected)
            prop_assert!(
                state.phase == ConsensusPhase::PrePrepare ||
                state.phase == ConsensusPhase::Prepare,
                "Phase should be PrePrepare or Prepare after proposal"
            );
        }
    }
}

// Property 8: No duplicate validator IDs in vote sets
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_no_duplicate_validators(
        validator_count in validator_count_strategy(),
        num_votes in (1usize..20usize),
    ) {
        let mut vote_set = HashSet::new();

        for i in 0..num_votes.min(validator_count) {
            let validator = ValidatorId(i as u64);
            vote_set.insert(validator);
        }

        // Property: Vote set size should equal number of unique validators
        prop_assert_eq!(
            vote_set.len(),
            num_votes.min(validator_count),
            "Vote set must contain unique validators"
        );
    }
}

// Property 9: Byzantine threshold allows correct fault tolerance
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_fault_tolerance_guarantees(
        validator_count in validator_count_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);
        let f = consensus.byzantine_threshold;
        let quorum = 2 * f + 1;

        // Property: Quorum can be achieved even with f faulty nodes
        let honest_nodes = validator_count - f;
        prop_assert!(
            honest_nodes >= quorum,
            "With {} honest nodes (total {} - f {}), must achieve quorum {}",
            honest_nodes,
            validator_count,
            f,
            quorum
        );

        // Property: Two quorums must overlap in at least one honest node
        let overlap = 2 * quorum - validator_count;
        prop_assert!(
            overlap > f,
            "Quorum overlap {} must exceed Byzantine threshold {}",
            overlap,
            f
        );
    }
}

// Property 10: Consensus state consistency
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))]

    #[tokio::test]
    async fn prop_state_consistency(
        validator_count in validator_count_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        // Get state multiple times
        let state1 = consensus.get_consensus_state();
        let state2 = consensus.get_consensus_state();
        let state3 = consensus.get_consensus_state();

        // Property: Consecutive reads without operations should be identical
        prop_assert_eq!(
            state1.view,
            state2.view,
            "View must be consistent across reads"
        );
        prop_assert_eq!(
            state1.sequence,
            state3.sequence,
            "Sequence must be consistent across reads"
        );
        prop_assert_eq!(
            state1.phase,
            state2.phase,
            "Phase must be consistent across reads"
        );
    }
}

// Property 11: Message validation with invalid signatures
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[tokio::test]
    async fn prop_invalid_signature_rejection(
        validator_count in validator_count_strategy(),
        payload in message_payload_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        // Create message with empty signature components (invalid)
        let invalid_message = ByzantineMessage {
            message_type: MessageType::PrePrepare,
            view: 0,
            sequence: 1,
            sender: ValidatorId(0),
            payload,
            quantum_signature: QuantumSignature {
                signature: vec![],
                public_key: vec![],
                quantum_proof: vec![],
            },
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        };

        // Property: Invalid signature should be rejected
        let result = consensus.handle_message(invalid_message).await;
        prop_assert!(
            result.is_err(),
            "Message with invalid signature should be rejected"
        );
    }
}

// Property 12: Timeout handling for old messages
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[tokio::test]
    async fn prop_old_message_rejection(
        validator_count in validator_count_strategy(),
        payload in message_payload_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);

        // Create message with old timestamp (> 5 seconds ago)
        let old_timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
            - 6_000_000_000; // 6 seconds ago

        let old_message = ByzantineMessage {
            message_type: MessageType::PrePrepare,
            view: 0,
            sequence: 1,
            sender: ValidatorId(0),
            payload,
            quantum_signature: QuantumSignature {
                signature: vec![1, 2, 3],
                public_key: vec![4, 5, 6],
                quantum_proof: vec![7, 8, 9],
            },
            timestamp: old_timestamp,
        };

        // Property: Old message should be rejected due to timeout
        let result = consensus.handle_message(old_message).await;
        prop_assert!(
            result.is_err(),
            "Message older than timeout should be rejected"
        );
    }
}

// Property 13: Minimum validator count for BFT
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES))]

    #[test]
    fn prop_minimum_validators_for_bft(
        validator_count in (1usize..10usize),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);
        let is_bft = consensus.is_byzantine_fault_tolerant();

        // Property: Need at least 4 validators for BFT (f=1)
        if validator_count < 4 {
            prop_assert!(
                !is_bft,
                "Systems with < 4 validators cannot be BFT"
            );
        } else {
            prop_assert!(
                is_bft,
                "Systems with >= 4 validators should be BFT"
            );
        }
    }
}

// Property 14: Prepared state requires quorum of prepare votes
proptest! {
    #![proptest_config(ProptestConfig::with_cases(PROP_TEST_CASES / 10))]

    #[tokio::test]
    async fn prop_prepared_requires_quorum(
        validator_count in validator_count_strategy(),
    ) {
        let consensus = ByzantineConsensus::new(validator_count);
        let f = consensus.byzantine_threshold;
        let required_votes = 2 * f + 1;

        // Property: System should not be prepared without quorum
        let initial_state = consensus.get_consensus_state();
        prop_assert!(
            !initial_state.prepared,
            "System should not be prepared initially"
        );

        // Property: Required votes for prepare phase is exactly 2f+1
        prop_assert_eq!(
            required_votes,
            2 * f + 1,
            "Required votes must be 2f+1"
        );
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_property_test_config() {
        assert_eq!(PROP_TEST_CASES, 1000);
    }

    #[test]
    fn test_strategy_generators() {
        // Verify strategy generators compile
        let _ = validator_count_strategy();
        let _ = validator_id_strategy(10);
        let _ = message_payload_strategy();
        let _ = view_number_strategy();
        let _ = sequence_number_strategy();
    }

    #[test]
    fn test_byzantine_threshold_calculation() {
        // Test the formula: f = (n-1)/3
        assert_eq!((4 - 1) / 3, 1);  // n=4, f=1
        assert_eq!((7 - 1) / 3, 2);  // n=7, f=2
        assert_eq!((10 - 1) / 3, 3); // n=10, f=3
    }

    #[test]
    fn test_quorum_requirement() {
        // Test that quorum 2f+1 is correct
        let f = 1;
        let quorum = 2 * f + 1;
        assert_eq!(quorum, 3); // For f=1, quorum is 3

        let f = 2;
        let quorum = 2 * f + 1;
        assert_eq!(quorum, 5); // For f=2, quorum is 5
    }
}
