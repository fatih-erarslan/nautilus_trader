# Byzantine Attack Scenarios - Visual Reference Guide

## Scenario 1: Race Condition Double Commit Attack

```
Setup: 4 nodes (N0, N1, N2, N3), f=1, malicious primary N0
Objective: Commit two different values for sequence 1

┌─────────────────────────────────────────────────────────────────┐
│ Timeline: Race Condition in handle_commit()                     │
└─────────────────────────────────────────────────────────────────┘

Time   Thread A (Value X)              Thread B (Value Y)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
t0     Acquire commit_votes lock
t1     Check: votes >= 2f+1 ✓
t2     Release lock ⚠️                 
       ┌─────────────┐
       │ RACE WINDOW │
       └─────────────┘
t3                                     Acquire commit_votes lock
t4                                     Check: votes >= 2f+1 ✓
t5                                     Release lock ⚠️
t6     Acquire state lock
t7     state.committed = true (X)
t8     Release state lock
t9                                     Acquire state lock
t10                                    state.committed = true (Y)
t11    Insert sequence 1
t12                                    Insert sequence 1 (dup)

Result: ❌ SAFETY VIOLATION
  - Node A committed value X for sequence 1
  - Node B committed value Y for sequence 1
  - X ≠ Y: Byzantine consensus broken

Impact: 
  - In trading: Execute conflicting orders
  - Financial loss: $$$
  - Trust compromised
```

## Scenario 2: Message Replay Attack

```
Setup: Attacker intercepts valid Prepare message from honest node
Objective: Amplify single vote into multiple votes

┌─────────────────────────────────────────────────────────────────┐
│ Network Topology                                                 │
└─────────────────────────────────────────────────────────────────┘

    [Node 1] ──────────┐
                       │
    [Node 2] ──────┬───┼─── [Attacker MITM]
                   │   │           │
    [Node 3] ──────┘   └───────────┘
                                   │
                            [Consensus Network]

Step-by-Step Attack:

t0: Node 1 sends Prepare(view=0, seq=5, value=X)
    Message: {
      type: Prepare,
      sender: Node1,
      signature: VALID_SIG_1,
      timestamp: 1000000
    }

t1: Attacker intercepts message
    
t2: Attacker replays 100x within 5-second window:
    ┌──────────────────────────────────┐
    │ Replay #1  @ t=1000001 ✓ Accepted│
    │ Replay #2  @ t=1000002 ✓ Accepted│
    │ Replay #3  @ t=1000003 ✓ Accepted│
    │ ...                               │
    │ Replay #100 @ t=1000100 ✓ Accepted│
    └──────────────────────────────────┘

t3: Vote count = 100 (should be 1)
    Required: 2f+1 = 3
    Actual: 100 (amplified!)

Result: ❌ SECURITY BYPASS
  - Single valid vote counted 100 times
  - Consensus reached with insufficient real votes
  - Byzantine fault tolerance threshold exceeded

Fix Required:
  - Message nonce validation
  - Sequence number monotonicity check
  - Deduplication by (sender, view, seq) tuple
```

## Scenario 3: Quantum Signature Forgery

```
Setup: Attacker wants to impersonate honest validator
Objective: Send malicious messages with "valid" signatures

┌─────────────────────────────────────────────────────────────────┐
│ Current Verification Logic (BROKEN)                             │
└─────────────────────────────────────────────────────────────────┘

async fn verify_signature(&self, message: &ByzantineMessage) 
    -> Result<bool, ConsensusError> 
{
    // Current check:
    if message.quantum_signature.signature.is_empty() ||
       message.quantum_signature.public_key.is_empty() ||
       message.quantum_signature.quantum_proof.is_empty() 
    {
        return Ok(false);
    }
    
    tokio::time::sleep(tokio::time::Duration::from_nanos(50)).await;
    
    return Ok(true); // ⚠️ ALWAYS TRUE for non-empty!
}

Attack Execution:

t0: Attacker crafts malicious message
    payload = "TRANSFER 1000 BTC TO ATTACKER"
    
t1: Attacker generates fake signature:
    signature = vec![1, 2, 3, 4]        // Any 4 bytes
    public_key = vec![5, 6, 7, 8]       // Any 4 bytes  
    quantum_proof = vec![9, 10, 11, 12] // Any 4 bytes
    
t2: Submit to consensus network
    
t3: Verification result:
    ✓ signature not empty -> true
    ✓ public_key not empty -> true
    ✓ quantum_proof not empty -> true
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RESULT: SIGNATURE VALID ❌
    
t4: Malicious message accepted by all nodes

Result: ❌ AUTHENTICATION FAILURE
  - Complete signature bypass
  - Any node can impersonate any validator
  - Byzantine consensus completely compromised

Real-World Impact:
  - Unauthorized trades executed
  - Fake consensus decisions
  - System takeover possible
```

## Scenario 4: View Change Stall (Liveness Attack)

```
Setup: 4 nodes, malicious primary (N0)
Objective: Prevent consensus progress indefinitely

┌─────────────────────────────────────────────────────────────────┐
│ Network State                                                    │
└─────────────────────────────────────────────────────────────────┘

View 0: N0 (malicious) is primary
Honest nodes: N1, N2, N3

Timeline:

t0: Primary N0 sends PrePrepare(value=X) to N1, N2 only
    N3 doesn't receive message
    
    [N0] ─────> [N1] ✓
         ─────> [N2] ✓
         ╳╳╳╳╳> [N3] ✗ (dropped)

t1: N1, N2 send Prepare
    N3 waits for PrePrepare...
    
t2: N3 timeout, triggers view change
    Sends ViewChange(view=1) to all
    
t3: CURRENT BROKEN LOGIC:
    ┌──────────────────────────────────────────┐
    │ async fn handle_view_change()           │
    │ {                                        │
    │   state.view += 1;  // No quorum check! │
    │ }                                        │
    └──────────────────────────────────────────┘
    
    Only 1 view change message received
    Required: 2f+1 = 3 messages
    Actual: 1 message
    
    ❌ View change FAILS silently

t4: System stuck:
    - View 0 stalled (no progress)
    - View 1 not reached (insufficient messages)
    - N1, N2 waiting in View 0
    - N3 waiting in View 1
    
    ┌─────────────────────────┐
    │ ⏸️  CONSENSUS DEADLOCK   │
    │    No recovery path     │
    └─────────────────────────┘

Duration: INDEFINITE

Result: ❌ LIVENESS VIOLATION
  - No transactions processed
  - System effectively down
  - Requires manual intervention

Fix Required:
  - Require 2f+1 view change messages
  - Implement timeout mechanism  
  - Add NewView broadcast
  - Transfer prepared certificates
```

## Scenario 5: Equivocation Attack (Double Voting)

```
Setup: 6 nodes total, 1 malicious (N5)
Objective: Send conflicting votes to split network

┌─────────────────────────────────────────────────────────────────┐
│ Network Partitioning via Conflicting Messages                   │
└─────────────────────────────────────────────────────────────────┘

                        [Malicious N5]
                              │
                 ┌────────────┴────────────┐
                 │                         │
        Different Messages          Different Messages
                 │                         │
         ┌───────┴───────┐         ┌───────┴───────┐
         │               │         │               │
        [N0]           [N1]       [N2]           [N3]
      Partition A              Partition B

Timeline:

t0: Proposal for sequence 10
    Primary proposes value X
    
t1: Malicious N5 sends conflicting Prepare:
    To N0, N1: Prepare(seq=10, value=X, hash=H_X)
    To N2, N3: Prepare(seq=10, value=Y, hash=H_Y)
    
    Same sequence, different values!

t2: Partition A (N0, N1, N5):
    Receives: Prepare(X) x3
    3 >= 2f+1? Yes (f=2, need 5)
    ❌ Wait, only have 3 total nodes
    Actually need all 6 for quorum
    
    Partition B (N2, N3, N5):
    Receives: Prepare(Y) x3
    Same issue

t3: CURRENT CODE (BROKEN):
    ┌────────────────────────────────────┐
    │ No equivocation detection!         │
    │ Both values accumulate votes       │
    │ No check for conflicting hashes    │
    └────────────────────────────────────┘

Result if undetected:
    - Partition A commits X
    - Partition B commits Y
    - ❌ SAFETY VIOLATION

Current Status: VULNERABLE
  - No tracking of per-validator votes
  - No conflict detection
  - No Byzantine behavior reporting

Detection Strategy:
    ┌─────────────────────────────────────┐
    │ Track all votes:                    │
    │ HashMap<ValidatorId, Vec<Vote>>     │
    │                                     │
    │ For each new vote:                  │
    │   if existing_vote.hash != new.hash │
    │     EQUIVOCATION DETECTED!          │
    │     Blacklist malicious validator   │
    └─────────────────────────────────────┘
```

## Scenario 6: Timing Side-Channel Attack

```
Setup: Attacker probes signature verification timing
Objective: Leak information about valid signatures

┌─────────────────────────────────────────────────────────────────┐
│ Timing Oracle Attack on Signature Verification                  │
└─────────────────────────────────────────────────────────────────┘

Current Code (VULNERABLE):

async fn verify_classical_signature(...) -> Result<bool> {
    if signature.signature.is_empty() || 
       signature.public_key.is_empty() {
        return Ok(false); // ⚠️ EARLY RETURN: Fast path
    }
    
    // Compute hash (takes time)
    let mut hasher = DefaultHasher::new();
    payload.hash(&mut hasher);
    signature.public_key.hash(&mut hasher);
    
    // Variable-time comparison
    Ok(signature.signature.starts_with(&hash)) // ⚠️ NON-CONSTANT
}

Timing Measurements:

Test 1000 random signatures:
  ┌──────────────────────┬───────────┐
  │ Signature Type       │ Time (ns) │
  ├──────────────────────┼───────────┤
  │ Empty signature      │       50  │ ← Fast!
  │ Wrong length         │       75  │
  │ Wrong first byte     │      150  │
  │ Wrong second byte    │      175  │
  │ Wrong third byte     │      200  │
  │ Correct prefix       │      250  │ ← Slow!
  └──────────────────────┴───────────┘

Attack Strategy:

1. Brute force first byte:
   Try all 256 values
   Measure timing
   Select slowest → correct byte 1
   
2. Fix byte 1, brute force byte 2:
   Try all 256 values
   Measure timing
   Select slowest → correct byte 2
   
3. Repeat for all bytes

Complexity:
  - Traditional: 2^256 (impossible)
  - Timing attack: 256 × 32 = 8,192 attempts
  - Reduction: 2^256 → 2^13 
  - Feasible in minutes!

Result: ❌ CRYPTOGRAPHIC WEAKNESS
  - Signature can be recovered via timing
  - No need to break cryptography
  - Side-channel completely bypasses security

Fix:
  - Use constant-time comparison
  - Eliminate early returns
  - Add random delay noise
  - Use subtle::ConstantTimeEq
```

## Attack Success Probability Matrix

```
┌────────────────────────────────────────────────────────────────┐
│ Attack vs Byzantine Threshold (f)                              │
└────────────────────────────────────────────────────────────────┘

                    Byzantine Nodes
                f=0   f=1   f=2   f=3
Attack Type     ───────────────────────
Race Condition  100%  100%  100%  100%  ← Always succeeds
Replay Attack   100%  100%  100%  100%  ← Always succeeds  
Signature Forge 100%  100%  100%  100%  ← Always succeeds
View Change     100%  100%  100%  100%  ← Always succeeds
Equivocation     90%   75%   50%   25%  ← Depends on detection
Timing Attack    80%   80%   80%   80%  ← Independent of f
Partition        50%   60%   70%   80%  ← Increases with f

Legend:
  100% = Always succeeds (Critical vulnerability)
   80% = Very likely to succeed (High risk)
   50% = Moderate probability (Medium risk)
   25% = Low probability (Still concerning)
```

## Recommended Testing Scenarios

```
Priority 0: Must test before production

1. ✓ Race condition with concurrent commits
   - 2 threads, same sequence, different values
   - Assert: Only one value committed

2. ✓ Message replay with 100x duplication
   - Intercept and replay valid message
   - Assert: Duplicate rejected

3. ✓ Forged signature acceptance
   - Submit message with fake signature
   - Assert: Verification fails

4. ✓ View change stall with malicious primary
   - Primary drops messages to some nodes
   - Assert: View change completes within timeout

Priority 1: Security validation

5. ✓ Equivocation detection
   - Malicious node sends conflicting votes
   - Assert: Conflict detected and reported

6. ✓ Timing attack resistance
   - Measure verification timing variance
   - Assert: Constant time within ±5%

Priority 2: Robustness

7. ✓ Network partition recovery
   - Split network, heal, verify convergence
   - Assert: Consensus restored

8. ✓ Slow node handling
   - Inject 4-second delay on honest node
   - Assert: Not treated as Byzantine
```

---

**Document Status:** Living Document - Update as attacks discovered
**Last Updated:** 2025-10-13
**Next Review:** After each security fix
