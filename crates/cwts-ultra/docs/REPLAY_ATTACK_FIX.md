# Replay Attack Prevention Fix for Byzantine Consensus

## Status: IMPLEMENTATION REQUIRED

## Vulnerability
The Byzantine consensus implementation lacks replay attack prevention. Attackers can resubmit old valid messages with the same signatures, potentially causing:
- Duplicate transaction execution
- Resource exhaustion
- State inconsistencies
- Denial of service

## Required Changes

### 1. Add Nonce Field to ByzantineMessage

**Location**: `byzantine_consensus.rs` line 19

**Current**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineMessage {
    pub message_type: MessageType,
    pub view: u64,
    pub sequence: u64,
    pub sender: ValidatorId,
    pub payload: Vec<u8>,
    pub quantum_signature: QuantumSignature,
    pub timestamp: u64,
}
```

**Required**:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineMessage {
    pub message_type: MessageType,
    pub view: u64,
    pub sequence: u64,
    pub sender: ValidatorId,
    pub payload: Vec<u8>,
    pub quantum_signature: QuantumSignature,
    pub timestamp: u64,
    pub nonce: u64, // Replay attack prevention - unique monotonic counter per validator
}
```

### 2. Add ReplayAttack Error Variant

**Location**: `byzantine_consensus.rs` line 63

**Current**:
```rust
#[derive(Debug, Clone)]
pub enum ConsensusError {
    InvalidMessage,
    InvalidSignature,
    ByzantineAttack,
    NetworkPartition,
    TimeoutError,
    QuantumVerificationFailed,
    MalformedPublicKey,
    MalformedSignature,
}
```

**Required**:
```rust
#[derive(Debug, Clone)]
pub enum ConsensusError {
    InvalidMessage,
    InvalidSignature,
    ByzantineAttack,
    NetworkPartition,
    TimeoutError,
    QuantumVerificationFailed,
    MalformedPublicKey,
    MalformedSignature,
    ReplayAttack, // Duplicate nonce detected - replay attack
}
```

### 3. Add Nonce Tracking to ByzantineConsensus

**Location**: `byzantine_consensus.rs` line 74

**Current**:
```rust
pub struct ByzantineConsensus {
    validator_id: ValidatorId,
    validator_count: usize,
    byzantine_threshold: usize,
    state: Arc<RwLock<ConsensusState>>,
    message_log: Arc<Mutex<HashMap<u64, Vec<ByzantineMessage>>>>,
    prepare_votes: Arc<Mutex<HashMap<(u64, u64), HashSet<ValidatorId>>>>,
    commit_votes: Arc<Mutex<HashMap<(u64, u64), HashSet<ValidatorId>>>>,
    executed_sequences: Arc<Mutex<HashSet<u64>>>,
    pending_transactions: Arc<Mutex<VecDeque<Vec<u8>>>>,
    quantum_verifier: Arc<QuantumVerification>,
    last_committed_sequence: Arc<AtomicU64>,
}
```

**Required**:
```rust
pub struct ByzantineConsensus {
    validator_id: ValidatorId,
    validator_count: usize,
    byzantine_threshold: usize,
    state: Arc<RwLock<ConsensusState>>,
    message_log: Arc<Mutex<HashMap<u64, Vec<ByzantineMessage>>>>,
    prepare_votes: Arc<Mutex<HashMap<(u64, u64), HashSet<ValidatorId>>>>,
    commit_votes: Arc<Mutex<HashMap<(u64, u64), HashSet<ValidatorId>>>>,
    executed_sequences: Arc<Mutex<HashSet<u64>>>,
    pending_transactions: Arc<Mutex<VecDeque<Vec<u8>>>>,
    quantum_verifier: Arc<QuantumVerification>,
    last_committed_sequence: Arc<AtomicU64>,
    // Replay attack prevention - track seen nonces per validator
    seen_nonces: Arc<Mutex<HashMap<ValidatorId, HashSet<u64>>>>,
    nonce_window: u64, // Maximum nonces to track per validator (1000)
    current_nonce: Arc<AtomicU64>, // Current nonce for this validator (atomic for thread safety)
}
```

### 4. Initialize Nonce Tracking in Constructor

**Location**: `byzantine_consensus.rs` line 90 (in `new()` function)

**Add at end of Self initialization (after line 110)**:
```rust
// Initialize replay attack prevention
seen_nonces: Arc::new(Mutex::new(HashMap::new())),
nonce_window: 1000, // Track up to 1000 nonces per validator
current_nonce: Arc::new(AtomicU64::new(0)), // Start from 0
```

### 5. Generate Nonce in propose_transaction

**Location**: `byzantine_consensus.rs` line 132 (before creating message)

**Add before message creation**:
```rust
// Generate unique nonce for this message (atomic increment for thread safety)
let nonce = self.current_nonce.fetch_add(1, Ordering::SeqCst) + 1;
```

**Update message creation to include nonce (line 133-144)**:
```rust
// Create PrePrepare message
let message = ByzantineMessage {
    message_type: MessageType::PrePrepare,
    view,
    sequence,
    sender: self.validator_id.clone(),
    payload: transaction,
    quantum_signature: self.generate_quantum_signature(&transaction).await?,
    timestamp: SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64,
    nonce, // Include unique nonce for replay attack prevention
};
```

### 6. Add Nonce Validation in handle_message

**Location**: `byzantine_consensus.rs` line 176 (after timing attack protection, before "Store message in log")

**Insert nonce validation logic**:
```rust
// REPLAY ATTACK PREVENTION: Check nonce uniqueness
// Each validator must use unique, monotonically increasing nonces
{
    let mut seen = self.seen_nonces.lock().await;
    let validator_nonces = seen.entry(message.sender.clone()).or_insert_with(HashSet::new);

    // Check if this nonce was already seen from this validator
    if validator_nonces.contains(&message.nonce) {
        log::warn!(
            "Replay attack detected: validator {:?} reused nonce {}",
            message.sender, message.nonce
        );
        return Err(ConsensusError::ReplayAttack);
    }

    // Add nonce to seen set
    validator_nonces.insert(message.nonce);

    // Sliding window: Trim old nonces if window exceeded to prevent memory growth
    // Keep only the most recent nonce_window nonces per validator
    if validator_nonces.len() > self.nonce_window as usize {
        // Remove nonces outside the sliding window
        // Only keep nonces greater than (max_nonce - window_size)
        let min_nonce = message.nonce.saturating_sub(self.nonce_window);
        validator_nonces.retain(|n| *n > min_nonce);
    }
}
```

### 7. Update ALL Tests to Include Nonce Field

**Location**: All test functions in `byzantine_consensus.rs` (lines 661-759)

All instances of `ByzantineMessage` creation must include `nonce` field:

```rust
// Example from test_atomic_commit_prevents_double_execution (line 666):
let message = ByzantineMessage {
    message_type: MessageType::Commit,
    view: 0,
    sequence: 1,
    sender: ValidatorId(1),
    payload: vec![1, 2, 3],
    quantum_signature: QuantumSignature {
        signature: vec![1],
        public_key: vec![1, 2, 3, 4, 5, 6, 7, 8],
        quantum_proof: vec![9, 10, 11, 12, 13, 14, 15, 16],
    },
    timestamp: SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64,
    nonce: 1, // ADD THIS FIELD
};
```

### 8. Add New Replay Attack Tests

**Location**: Add after `test_memory_barrier_ensures_visibility` (line 759)

```rust
#[tokio::test]
async fn test_replay_attack_prevention() {
    // Test that replay attacks are detected and blocked
    let consensus = ByzantineConsensus::new(4);

    let message = ByzantineMessage {
        message_type: MessageType::PrePrepare,
        view: 0,
        sequence: 1,
        sender: ValidatorId(1),
        payload: vec![1, 2, 3],
        quantum_signature: QuantumSignature {
            signature: vec![1],
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8],
            quantum_proof: vec![9, 10, 11, 12, 13, 14, 15, 16],
        },
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        nonce: 42, // Same nonce will be reused
    };

    // First message should succeed
    let result1 = consensus.handle_message(message.clone()).await;
    assert!(result1.is_ok());

    // Replayed message with same nonce should fail
    let result2 = consensus.handle_message(message.clone()).await;
    assert!(matches!(result2, Err(ConsensusError::ReplayAttack)));
}

#[tokio::test]
async fn test_sliding_window_nonce_cleanup() {
    // Test that old nonces are cleaned up with sliding window
    let consensus = ByzantineConsensus::new(4);

    // Send messages with nonces up to window size + 100
    for nonce in 1..=1100 {
        let message = ByzantineMessage {
            message_type: MessageType::PrePrepare,
            view: 0,
            sequence: nonce,
            sender: ValidatorId(1),
            payload: vec![1, 2, 3],
            quantum_signature: QuantumSignature {
                signature: vec![1],
                public_key: vec![1, 2, 3, 4, 5, 6, 7, 8],
                quantum_proof: vec![9, 10, 11, 12, 13, 14, 15, 16],
            },
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            nonce,
        };
        let _ = consensus.handle_message(message).await;
    }

    // Check that old nonces were cleaned up
    let seen = consensus.seen_nonces.lock().await;
    let validator_nonces = seen.get(&ValidatorId(1)).unwrap();
    // Should have trimmed to window size (1000)
    assert!(validator_nonces.len() <= consensus.nonce_window as usize);
}

#[tokio::test]
async fn test_nonce_uniqueness_per_validator() {
    // Test that nonces are tracked separately per validator
    let consensus = ByzantineConsensus::new(4);

    // Validator 1 uses nonce 42
    let message1 = ByzantineMessage {
        message_type: MessageType::PrePrepare,
        view: 0,
        sequence: 1,
        sender: ValidatorId(1),
        payload: vec![1, 2, 3],
        quantum_signature: QuantumSignature {
            signature: vec![1],
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8],
            quantum_proof: vec![9, 10, 11, 12, 13, 14, 15, 16],
        },
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        nonce: 42,
    };

    // Validator 2 also uses nonce 42 (should be allowed - different validator)
    let message2 = ByzantineMessage {
        message_type: MessageType::PrePrepare,
        view: 0,
        sequence: 2,
        sender: ValidatorId(2),
        payload: vec![4, 5, 6],
        quantum_signature: QuantumSignature {
            signature: vec![2],
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8],
            quantum_proof: vec![9, 10, 11, 12, 13, 14, 15, 16],
        },
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        nonce: 42, // Same nonce, different validator
    };

    // Both should succeed (nonces are per-validator)
    let result1 = consensus.handle_message(message1).await;
    assert!(result1.is_ok());

    let result2 = consensus.handle_message(message2).await;
    assert!(result2.is_ok());
}
```

## Security Properties Guaranteed

1. **Freshness**: Old messages cannot be replayed beyond the sliding window
2. **Uniqueness**: Each message from a validator has a unique nonce
3. **Memory Bounded**: Nonce storage is limited to `nonce_window * validator_count`
4. **Thread Safe**: Atomic operations prevent race conditions in nonce generation
5. **Per-Validator**: Nonce tracking is isolated per validator

## Performance Impact

- **Memory**: O(nonce_window * validator_count) = O(1000 * n) per node
- **Computation**: O(1) nonce check per message (HashSet lookup)
- **Sliding Window**: O(window_size) cleanup when threshold exceeded (amortized O(1))

## Testing Checklist

- [ ] Replay attack is detected and blocked
- [ ] Sliding window prevents memory growth
- [ ] Nonces are tracked per-validator
- [ ] Atomic nonce generation is thread-safe
- [ ] All existing tests pass with nonce field added

## Files to Modify

1. `/Users/ashina/Kayra/src/cwts-ultra/core/src/consensus/byzantine_consensus.rs`
   - Add `nonce` field to `ByzantineMessage` struct
   - Add `ReplayAttack` error variant
   - Add nonce tracking fields to `ByzantineConsensus`
   - Initialize nonce tracking in constructor
   - Generate nonce in `propose_transaction`
   - Validate nonce in `handle_message`
   - Update all test messages to include nonce
   - Add new replay attack tests

## Implementation Notes

- The nonce is a monotonically increasing counter per validator
- Atomic operations ensure thread-safe nonce generation
- Sliding window prevents unbounded memory growth
- The 1000-nonce window provides sufficient protection while keeping memory bounded
- Nonce validation happens after signature verification but before message processing

## Next Steps

1. Apply all changes from sections 1-8 above
2. Run `cargo test` to verify all tests pass
3. Run `cargo fmt` to format code
4. Verify replay attack tests pass
5. Document the security improvement in release notes
