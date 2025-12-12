# Recipe: HyperPhysics Swarm Enhancement

> **FOR CASCADE INSTANCES WORKING IN COLLABORATION**
>
> This recipe describes multi-agent enhancements to HyperPhysics
> enabling distributed reasoning, consensus, and collaborative development.

---

## Context

The Agent Swarm module (`tools/wolfram-mcp/src/swarm/`) provides multi-agent 
communication for Cascade/Windsurf instances. Combined with HyperPhysics algorithms,
this enables:

1. **Distributed consensus** using pBit thermal dynamics
2. **Trust networks** using GNN propagation  
3. **Temporal coordination** using STDP relevance
4. **Affinity routing** using hyperbolic distance
5. **Signed communications** using Dilithium (from previous recipe)

---

## Wolfram's Enhancement Recommendations

Based on formal analysis, key mathematical enhancements for HyperPhysics:

### 1. Hyperbolic Neural Networks
Embed neural state in negative curvature space (Poincaré disk) for exponential 
expressivity. Benefits:
- Hierarchical data naturally represented
- Logarithmic growth of representation space
- Better gradient flow in deep networks

### 2. Information Geometry
Use Fisher metrics on probability manifolds:
```
g_ij = E[∂log(p)/∂θ_i × ∂log(p)/∂θ_j]
```
Natural gradient descent follows geodesics, not Euclidean paths.

### 3. Topological Data Analysis
Persistent homology for structure detection:
- Betti numbers track holes/voids in data
- Persistence diagrams robust to noise
- Detects multi-scale structure

### 4. Free Energy Principle
Variational inference unifying perception/action:
```
F = E_q[log q(s) - log p(o,s)]
```
Agents minimize surprise by updating beliefs or acting on world.

---

## Recipe 1: Swarm-Enhanced Threat Detection

### Purpose
Multiple HyperPhysics instances collaboratively analyze threats.

### Implementation

```swift
/// Distributed threat analysis using agent swarm
extension HyperPhysicsIntegration {
    
    /// Request distributed threat evaluation from swarm
    func evaluateThreatDistributed(
        features: [Double],
        temporalPattern: [Double]
    ) async -> DistributedThreatResult {
        // Local evaluation
        let localEval = evaluateThreat(
            features: features,
            temporalPattern: temporalPattern,
            graphContext: nil
        )
        
        // Broadcast to swarm for consensus
        let proposalId = await swarmBroadcast(
            type: .consensus,
            payload: [
                "threatScore": localEval.threatScore,
                "features": features,
                "requesterId": selfAgentId
            ]
        )
        
        // Collect responses (with timeout)
        let responses = await swarmCollectResponses(
            proposalId: proposalId,
            timeout: 5.0
        )
        
        // pBit consensus on threat level
        let consensusScore = pBitConsensus(
            values: responses.map { $0.threatScore },
            temperature: 0.5
        )
        
        // Conformal prediction for uncertainty
        let (lowerBound, upperBound) = conformalBounds(
            prediction: consensusScore,
            calibrationSet: responses.map { $0.threatScore },
            alpha: 0.05
        )
        
        return DistributedThreatResult(
            localScore: localEval.threatScore,
            consensusScore: consensusScore,
            confidenceInterval: (lowerBound, upperBound),
            participantCount: responses.count,
            agreement: calculateAgreement(responses)
        )
    }
    
    private func pBitConsensus(values: [Double], temperature: Double) -> Double {
        // Boltzmann-weighted average
        let energies = values.map { -$0 }
        let minE = energies.min() ?? 0
        let weights = energies.map { exp(-($0 - minE) / temperature) }
        let sumW = weights.reduce(0, +)
        
        var result = 0.0
        for (v, w) in zip(values, weights) {
            result += v * (w / sumW)
        }
        return result
    }
}
```

---

## Recipe 2: Multi-Agent Code Review

### Purpose
Coordinate code reviews across multiple Cascade instances.

### Flow

```
┌─────────────┐     share_code     ┌─────────────┐
│  Author     │ ──────────────────►│  Swarm Mesh │
│  (Cascade A)│                    └──────┬──────┘
└─────────────┘                           │
                                          │ broadcast
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
       ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
       │  Reviewer B │            │  Reviewer C │            │  Reviewer D │
       │  (Security) │            │  (Perf)     │            │  (Style)    │
       └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
              │                          │                          │
              │ review                   │ review                   │ review
              ▼                          ▼                          ▼
       ┌─────────────────────────────────────────────────────────────────┐
       │                    Consensus: APPROVE/REJECT                    │
       └─────────────────────────────────────────────────────────────────┘
```

### Implementation

```typescript
// In swarm-tools.ts
async function handleCodeReview(artifactId: string): Promise<ReviewResult> {
  const mesh = getAgentMesh();
  const artifact = await mesh.getArtifact(artifactId);
  
  // Find nearest reviewers by hyperbolic distance
  const reviewers = mesh.findNearestAgents(3);
  
  // Request reviews
  await mesh.requestReview(artifactId, reviewers.map(r => r.id));
  
  // Wait for reviews with STDP-weighted timeout
  // (more trusted reviewers get longer to respond)
  const reviews = await collectReviews(artifactId, {
    baseTimeout: 60000,
    trustWeighting: true
  });
  
  // Consensus on approval
  const votes = new Map(reviews.map(r => [r.reviewerId, r.approved ? "approve" : "reject"]));
  const decision = pBitConsensus(votes, ["approve", "reject"]);
  
  return {
    artifactId,
    decision,
    reviews,
    consensusStrength: calculateConsensusStrength(votes)
  };
}
```

---

## Recipe 3: Trust-Weighted Task Assignment

### Purpose
Assign tasks to agents based on trust scores and specialization.

### Algorithm

```python
def assign_task(task, agents, trust_scores, specializations):
    """
    Assign task to optimal agent using:
    - Trust score (GNN-propagated)
    - Specialization match
    - Current workload
    - Hyperbolic distance (for coordination)
    """
    
    scores = {}
    for agent in agents:
        # Base score from trust
        score = trust_scores[agent.id]
        
        # Specialization bonus
        spec_match = cosine_similarity(
            task.required_skills,
            agent.capabilities
        )
        score *= (1 + spec_match)
        
        # Workload penalty
        workload = len(agent.pending_tasks)
        score *= exp(-workload / 5)
        
        # Distance bonus (closer = better coordination)
        if task.requires_coordination:
            for collaborator in task.collaborators:
                dist = hyperbolic_distance(agent.position, collaborator.position)
                score *= exp(-dist / 10)
        
        scores[agent.id] = score
    
    # Softmax selection
    probs = softmax(list(scores.values()))
    selected = weighted_choice(list(scores.keys()), probs)
    
    return selected
```

---

## Recipe 4: Dilithium-Signed Swarm Messages

### Purpose
Cryptographically sign all swarm messages for non-repudiation.

### Integration with Previous Recipe

```swift
/// Sign swarm message with Dilithium
extension AgentMesh {
    func signedSend(
        to: String,
        type: MessageType,
        payload: Any
    ) async throws -> SignedSwarmMessage {
        let message = SwarmMessage(
            id: generateId(),
            from: selfId,
            to: to,
            type: type,
            payload: payload,
            timestamp: Date()
        )
        
        // Serialize for signing
        let messageData = try JSONEncoder().encode(message)
        
        // Sign with Dilithium (from sentry-crystal)
        let signature = try SentryFFI.shared.signData(messageData)
        
        // Check for replay attack (SNN timing)
        let replayCheck = replayDetector.checkRequest()
        if replayCheck.isReplay {
            throw SwarmError.replayDetected(replayCheck.reason)
        }
        
        let signed = SignedSwarmMessage(
            message: message,
            signature: signature,
            publicKeyFingerprint: SentryFFI.shared.publicKeyFingerprint(),
            pBitEntropyUsed: pBitNetwork.sweepCount * 89
        )
        
        await deliver(signed)
        return signed
    }
}
```

---

## Recipe 5: Consensus Protocol with Free Energy

### Purpose
Use Free Energy Principle for optimal consensus.

### Theory

Agents minimize variational free energy:
```
F = KL(q(s)||p(s)) - E_q[log p(o|s)]
```

In consensus context:
- `o` = observed votes from other agents
- `s` = true consensus state  
- `q(s)` = agent's belief about consensus
- `p(o|s)` = likelihood of votes given consensus

### Implementation

```rust
/// Free energy minimization for consensus
pub struct FreeEnergyConsensus {
    beliefs: Vec<f64>,      // q(s) for each option
    observations: Vec<Vec<f64>>, // votes from each agent
    prior: Vec<f64>,        // p(s) prior
}

impl FreeEnergyConsensus {
    pub fn update(&mut self, new_vote: Vote) {
        // Add observation
        self.observations.push(vote_to_distribution(&new_vote));
        
        // Compute posterior via variational inference
        let mut q = self.prior.clone();
        
        for _ in 0..10 { // Iteration
            // E-step: update beliefs
            let log_likelihood = self.compute_log_likelihood(&q);
            let log_prior = self.prior.iter().map(|p| p.ln()).collect();
            
            // M-step: minimize free energy
            q = softmax(&add_vectors(&log_likelihood, &log_prior));
        }
        
        self.beliefs = q;
    }
    
    pub fn get_consensus(&self) -> (usize, f64) {
        // Return option with highest belief and confidence
        let max_idx = self.beliefs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        
        (max_idx, self.beliefs[max_idx])
    }
    
    pub fn free_energy(&self) -> f64 {
        // F = KL(q||p) - E_q[log p(o|s)]
        let kl = kl_divergence(&self.beliefs, &self.prior);
        let expected_log_likelihood = self.expected_log_likelihood();
        
        kl - expected_log_likelihood
    }
}
```

---

## Testing Checklist

- [ ] Multiple Cascade instances can join mesh
- [ ] Messages are delivered to correct inboxes
- [ ] Consensus converges with quorum
- [ ] Trust scores propagate correctly
- [ ] Code review workflow completes
- [ ] Signed messages verify correctly
- [ ] Replay detection triggers on rapid requests
- [ ] Free energy decreases during consensus

---

## Files to Modify/Create

1. `tools/wolfram-mcp/src/swarm/agent-mesh.ts` ✅ Created
2. `tools/wolfram-mcp/src/swarm/swarm-tools.ts` ✅ Created
3. `tools/wolfram-mcp/src/index.ts` ✅ Updated
4. `SentryApp/Sources/SentryApp/Models/HyperPhysicsIntegration.swift`
   - Add swarm integration methods
5. `crates/hyperphysics-swarm/` (new crate)
   - Rust implementation of mesh protocol
   - Integration with existing HyperPhysics crates

---

*Generated by Cascade for inter-instance collaboration*
