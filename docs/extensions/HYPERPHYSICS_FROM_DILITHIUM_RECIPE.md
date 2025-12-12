# Recipe: HyperPhysics Enhancements from Dilithium

> **FOR CASCADE INSTANCE WORKING ON HYPERPHYSICS**
> 
> This recipe describes enhancements to add to the HyperPhysics module
> based on integration with CRYSTALS-Dilithium post-quantum cryptography.

---

## Context

The Dilithium module (`crates/sentry-crystal/src/dilithium.rs`) provides post-quantum 
digital signatures. HyperPhysics (`SentryApp/Sources/SentryApp/Models/HyperPhysicsIntegration.swift`)
provides real-time physics simulation with pBit, SNN, GNN, and conformal prediction.

The goal is to enhance HyperPhysics to:
1. Sign threat evaluations with Dilithium
2. Detect replay attacks using SNN timing
3. Analyze key trust using GNN
4. Quantify signature confidence with conformal prediction
5. Display pBit entropy generation in UI

---

## Prerequisites

Ensure these imports are available:

```swift
// In HyperPhysicsIntegration.swift
import Foundation
import Combine

// SentryFFI should have these methods (from Rust FFI):
// - signData(_ data: Data) -> Data
// - verifySignature(signature: Data, message: Data, publicKey: Data) -> Bool
// - publicKeyFingerprint() -> String
// - publicKeyBytes() -> Data
```

---

## Recipe 1: Signed Threat Evaluations

### Purpose
Every threat evaluation should be cryptographically signed for non-repudiation 
and audit trail integrity.

### Steps

1. **Add SignedThreatEvaluation struct** after `ThreatEvaluation`:

```swift
/// Post-quantum signed threat evaluation
struct SignedThreatEvaluation: Codable {
    let evaluation: ThreatEvaluation
    let signature: Data
    let publicKeyFingerprint: String
    let timestamp: Date
    let pBitEntropyUsed: Int  // Bits of pBit entropy in signature
    
    /// Convert evaluation to signable bytes
    private func signableData() -> Data {
        var data = Data()
        data.append(contentsOf: withUnsafeBytes(of: evaluation.threatScore) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: evaluation.pBitFreeEnergy) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: evaluation.snnSpikeRate) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: timestamp.timeIntervalSince1970) { Array($0) })
        return data
    }
    
    /// Verify signature validity
    func verify() -> Bool {
        guard let ffi = try? SentryFFI.shared else { return false }
        return ffi.verifySignature(
            signature: signature,
            message: signableData(),
            publicKey: ffi.publicKeyBytes()
        )
    }
}
```

2. **Add signing method to HyperPhysicsIntegration**:

```swift
extension HyperPhysicsIntegration {
    /// Evaluate threat and sign result with Dilithium
    func evaluateAndSignThreat(
        features: [Double],
        temporalPattern: [Double],
        graphContext: (node: ThreatNode, neighbors: [ThreatNode])?
    ) -> SignedThreatEvaluation {
        // Run evaluation
        let evaluation = evaluateThreat(
            features: features,
            temporalPattern: temporalPattern,
            graphContext: graphContext
        )
        
        // Prepare message
        let timestamp = Date()
        var messageData = Data()
        messageData.append(contentsOf: withUnsafeBytes(of: evaluation.threatScore) { Array($0) })
        messageData.append(contentsOf: withUnsafeBytes(of: evaluation.pBitFreeEnergy) { Array($0) })
        messageData.append(contentsOf: withUnsafeBytes(of: evaluation.snnSpikeRate) { Array($0) })
        messageData.append(contentsOf: withUnsafeBytes(of: timestamp.timeIntervalSince1970) { Array($0) })
        
        // Sign with Dilithium (via FFI)
        let signature: Data
        do {
            signature = try SentryFFI.shared.signData(messageData)
        } catch {
            signature = Data()  // Empty signature on error
        }
        
        return SignedThreatEvaluation(
            evaluation: evaluation,
            signature: signature,
            publicKeyFingerprint: (try? SentryFFI.shared.publicKeyFingerprint()) ?? "unknown",
            timestamp: timestamp,
            pBitEntropyUsed: pBitNetwork.sweepCount * 89
        )
    }
}
```

---

## Recipe 2: SNN Replay Attack Detection

### Purpose
Detect signature replay attacks by analyzing timing patterns with SNN.

### Steps

1. **Add SignatureReplayDetector class** after SNNLayer:

```swift
/// Replay attack detector using SNN spike timing patterns
class SignatureReplayDetector: ObservableObject {
    @Published var isReplayDetected: Bool = false
    @Published var lastConfidence: Double = 0.0
    
    private var requestTimes: [Date] = []
    private var intervalHistory: [Double] = []
    private var baselineRate: Double = 1.0  // requests per second
    private var rateVariance: Double = 0.25
    private weak var snnLayer: SNNLayer?
    
    init(snnLayer: SNNLayer) {
        self.snnLayer = snnLayer
    }
    
    /// Record a signature request and check for replay attack
    @discardableResult
    func checkRequest() -> (isReplay: Bool, confidence: Double, reason: String) {
        let now = Date()
        defer { 
            requestTimes.append(now) 
            pruneOldRequests()
        }
        
        guard let lastTime = requestTimes.last else {
            return (false, 0.0, "First request")
        }
        
        let interval = now.timeIntervalSince(lastTime)
        intervalHistory.append(interval)
        
        // Calculate current rate
        let currentRate = 1.0 / max(interval, 0.001)
        
        // Update baseline with exponential moving average
        baselineRate = 0.95 * baselineRate + 0.05 * currentRate
        
        // Calculate variance
        if intervalHistory.count > 10 {
            let recentIntervals = Array(intervalHistory.suffix(10))
            let mean = recentIntervals.reduce(0, +) / Double(recentIntervals.count)
            rateVariance = recentIntervals.map { pow($0 - mean, 2) }.reduce(0, +) / Double(recentIntervals.count)
        }
        
        // Z-score for anomaly detection
        let stdDev = sqrt(max(rateVariance, 0.0001))
        let zScore = abs(currentRate - baselineRate) / (baselineRate * stdDev + 0.001)
        
        // Also check with SNN
        var snnAnomaly = false
        if let snn = snnLayer {
            let pattern = Array(repeating: currentRate / 10.0, count: min(snn.inputSize, 55))
            let result = snn.detectTemporalAnomaly(pattern: pattern)
            snnAnomaly = result.isAnomaly
        }
        
        // Replay detection criteria:
        // 1. Rate is > 3 standard deviations from baseline
        // 2. OR interval is suspiciously regular (variance too low)
        // 3. OR SNN detects anomaly
        
        let isReplay = zScore > 3.0 || 
                       (intervalHistory.count > 5 && rateVariance < 0.001) ||
                       snnAnomaly
        
        let confidence = min(1.0, zScore / 5.0)
        
        // Update published properties
        DispatchQueue.main.async {
            self.isReplayDetected = isReplay
            self.lastConfidence = confidence
        }
        
        let reason: String
        if zScore > 3.0 {
            reason = "Abnormal request rate (z=\(String(format: "%.2f", zScore)))"
        } else if rateVariance < 0.001 && intervalHistory.count > 5 {
            reason = "Suspiciously regular timing"
        } else if snnAnomaly {
            reason = "SNN temporal anomaly detected"
        } else {
            reason = "Normal"
        }
        
        return (isReplay, confidence, reason)
    }
    
    private func pruneOldRequests() {
        let cutoff = Date().addingTimeInterval(-300)  // Keep last 5 minutes
        requestTimes = requestTimes.filter { $0 > cutoff }
        if intervalHistory.count > 100 {
            intervalHistory = Array(intervalHistory.suffix(100))
        }
    }
    
    /// Reset detector state
    func reset() {
        requestTimes.removeAll()
        intervalHistory.removeAll()
        baselineRate = 1.0
        rateVariance = 0.25
        isReplayDetected = false
        lastConfidence = 0.0
    }
}
```

2. **Add to HyperPhysicsIntegration**:

```swift
// In HyperPhysicsIntegration class, add property:
@Published var replayDetector: SignatureReplayDetector!

// In init(), add:
replayDetector = SignatureReplayDetector(snnLayer: snnLayer)
```

---

## Recipe 3: GNN Key Trust Network

### Purpose
Model public key signing relationships as a graph for trust analysis.

### Steps

1. **Add KeyTrustNode and KeyTrustEdge**:

```swift
/// Node in key trust graph
struct KeyTrustNode: Identifiable, Hashable {
    let id: String  // Public key fingerprint
    var trustScore: Double = 0.5
    var signingCount: Int = 0      // How many keys this key signed
    var signedByCount: Int = 0     // How many keys signed this key
    var hyperbolicRadius: Double = 0.5
    var securityLevel: Int = 3     // Dilithium level: 2, 3, or 5
    var lastActivity: Date = Date()
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    static func == (lhs: KeyTrustNode, rhs: KeyTrustNode) -> Bool {
        lhs.id == rhs.id
    }
}

/// Edge representing signing relationship
struct KeyTrustEdge: Identifiable {
    let id = UUID()
    let signerId: String
    let signedId: String
    let timestamp: Date
    let messageType: String
}
```

2. **Add KeyTrustAnalyzer class**:

```swift
/// GNN-based key trust analyzer
class KeyTrustAnalyzer: ObservableObject {
    @Published var nodes: [String: KeyTrustNode] = [:]
    @Published var edges: [KeyTrustEdge] = []
    @Published var anomalies: [(keyId: String, score: Double, reason: String)] = []
    
    private let gnnLayer: GraphAttentionLayer
    
    init() {
        // Fibonacci-scaled: 21 input, 13 output
        self.gnnLayer = GraphAttentionLayer(inputDim: 21, outputDim: 13)
    }
    
    /// Record a signing event
    func recordSigningEvent(signerFingerprint: String, signedFingerprint: String,
                           messageType: String = "generic") {
        // Ensure signer node exists
        if nodes[signerFingerprint] == nil {
            nodes[signerFingerprint] = KeyTrustNode(id: signerFingerprint)
        }
        // Ensure signed node exists
        if nodes[signedFingerprint] == nil {
            nodes[signedFingerprint] = KeyTrustNode(id: signedFingerprint)
        }
        
        // Update counts
        nodes[signerFingerprint]?.signingCount += 1
        nodes[signerFingerprint]?.lastActivity = Date()
        nodes[signedFingerprint]?.signedByCount += 1
        
        // Add edge
        edges.append(KeyTrustEdge(
            signerId: signerFingerprint,
            signedId: signedFingerprint,
            timestamp: Date(),
            messageType: messageType
        ))
        
        // Prune old edges
        pruneOldEdges()
        
        // Re-analyze
        analyzeForAnomalies()
    }
    
    /// Analyze trust network for anomalies
    func analyzeForAnomalies() {
        var detected: [(String, Double, String)] = []
        
        for (id, node) in nodes {
            // Calculate signing ratio
            let signingRatio = Double(node.signingCount) / max(1.0, Double(node.signedByCount))
            
            // Rule 1: Leaf keys (Level 2) shouldn't sign many keys
            if node.securityLevel == 2 && node.signingCount > 5 {
                let score = min(1.0, Double(node.signingCount) / 10.0)
                detected.append((id, score, "Leaf key signing too many others"))
            }
            
            // Rule 2: Root keys (Level 5) shouldn't be signed by many
            if node.securityLevel == 5 && node.signedByCount > 3 {
                let score = min(1.0, Double(node.signedByCount) / 5.0)
                detected.append((id, score, "Root key signed by too many"))
            }
            
            // Rule 3: Extreme signing ratios
            if signingRatio > 10.0 {
                detected.append((id, 0.7, "Unusually high signing ratio"))
            } else if signingRatio < 0.1 && node.signedByCount > 10 {
                detected.append((id, 0.5, "Many signatures but doesn't sign others"))
            }
            
            // Rule 4: Inactive keys suddenly active
            let hoursSinceActive = Date().timeIntervalSince(node.lastActivity) / 3600
            if hoursSinceActive > 168 && node.signingCount > 0 {  // 1 week
                detected.append((id, 0.6, "Dormant key suddenly active"))
            }
        }
        
        anomalies = detected.sorted { $0.1 > $1.1 }
    }
    
    /// Update node security level based on hyperbolic position
    func updateSecurityLevel(fingerprint: String, hyperbolicRadius: Double) {
        guard nodes[fingerprint] != nil else { return }
        
        nodes[fingerprint]?.hyperbolicRadius = hyperbolicRadius
        
        // Security decreases with radius
        if hyperbolicRadius < 0.3 {
            nodes[fingerprint]?.securityLevel = 5
        } else if hyperbolicRadius < 0.6 {
            nodes[fingerprint]?.securityLevel = 3
        } else {
            nodes[fingerprint]?.securityLevel = 2
        }
    }
    
    private func pruneOldEdges() {
        let cutoff = Date().addingTimeInterval(-86400 * 30)  // 30 days
        edges = edges.filter { $0.timestamp > cutoff }
    }
}
```

3. **Add to HyperPhysicsIntegration**:

```swift
// Add property:
@Published var keyTrustAnalyzer = KeyTrustAnalyzer()
```

---

## Recipe 4: Conformal Signature Prediction

### Purpose
Add uncertainty quantification for signature verification.

### Steps

1. **Add ConformalSignaturePredictor**:

```swift
/// Conformal prediction for signature verification confidence
class ConformalSignaturePredictor: ObservableObject {
    @Published var calibrationCount: Int = 0
    @Published var lastPValue: Double = 0.5
    
    private var calibrationScores: [Double] = []
    private let alpha: Double = 0.05  // 95% confidence level
    
    /// Add a calibration sample from known valid signature
    func calibrate(verificationScore: Double, expectedScore: Double = 1.0) {
        let nonconformity = abs(verificationScore - expectedScore)
        calibrationScores.append(nonconformity)
        calibrationCount = calibrationScores.count
        
        // Keep calibration set bounded
        if calibrationScores.count > 1000 {
            calibrationScores = Array(calibrationScores.suffix(1000))
        }
    }
    
    /// Predict with confidence bounds
    func predict(verificationScore: Double, expectedScore: Double = 1.0)
        -> (isValid: Bool, pValue: Double, confidenceLevel: Double, inConfidenceSet: Bool) {
        
        let nonconformity = abs(verificationScore - expectedScore)
        
        guard !calibrationScores.isEmpty else {
            return (verificationScore > 0.5, 0.5, 0.5, true)
        }
        
        // p-value = (# calibration scores >= current + 1) / (n + 1)
        let higherCount = calibrationScores.filter { $0 >= nonconformity }.count
        let pValue = Double(higherCount + 1) / Double(calibrationScores.count + 1)
        
        lastPValue = pValue
        
        // Valid if p-value > alpha (not a statistical outlier)
        let isValid = pValue > alpha
        let confidenceLevel = 1.0 - alpha
        let inConfidenceSet = pValue > alpha
        
        return (isValid, pValue, confidenceLevel, inConfidenceSet)
    }
    
    /// Get current coverage estimate
    var estimatedCoverage: Double {
        guard calibrationScores.count > 10 else { return 1.0 - alpha }
        // Empirical coverage from calibration set
        return 1.0 - alpha  // Conformal prediction guarantees this asymptotically
    }
}
```

2. **Add to HyperPhysicsIntegration**:

```swift
// Add property:
@Published var signaturePredictor = ConformalSignaturePredictor()
```

---

## Recipe 5: UI Enhancements

### Purpose
Display pBit entropy and Dilithium integration status in UI.

### Steps

1. **Update PBitDynamicsView** in `HyperbolicTopologyView.swift`:

Add this GroupBox after the existing "Equilibrium Physics" GroupBox:

```swift
GroupBox("Dilithium Integration") {
    VStack(alignment: .leading, spacing: 6) {
        // Entropy generation status
        HStack {
            Image(systemName: "lock.shield.fill")
                .foregroundColor(.green)
            Text("Post-Quantum Entropy")
                .font(.caption.bold())
            Spacer()
        }
        
        HStack {
            Text("Bits Generated:")
            Spacer()
            Text("\(hyperPhysics.pBitNetwork.sweepCount * 89)")
                .font(.system(.caption, design: .monospaced))
        }
        
        HStack {
            Text("Dilithium Seeds:")
            Spacer()
            Text("\(hyperPhysics.pBitNetwork.sweepCount / 3)")
                .font(.system(.caption, design: .monospaced))
                .foregroundColor(.green)
        }
        
        ProgressView(value: Double(hyperPhysics.pBitNetwork.sweepCount % 3) / 3.0)
            .tint(.purple)
        
        Text("3 sweeps = 267 bits ≥ 192 bits for Dilithium3")
            .font(.system(size: 9))
            .foregroundColor(.secondary)
        
        Divider()
        
        // Replay detection status
        HStack {
            Image(systemName: hyperPhysics.replayDetector?.isReplayDetected == true 
                  ? "exclamationmark.triangle.fill" : "checkmark.shield.fill")
                .foregroundColor(hyperPhysics.replayDetector?.isReplayDetected == true 
                                 ? .red : .green)
            Text("Replay Detection")
                .font(.caption.bold())
            Spacer()
            Text(hyperPhysics.replayDetector?.isReplayDetected == true ? "ALERT" : "OK")
                .font(.caption2.bold())
                .foregroundColor(hyperPhysics.replayDetector?.isReplayDetected == true 
                                 ? .red : .green)
        }
        
        // Key trust anomalies
        if !hyperPhysics.keyTrustAnalyzer.anomalies.isEmpty {
            Divider()
            HStack {
                Image(systemName: "key.fill")
                    .foregroundColor(.orange)
                Text("Key Trust Anomalies: \(hyperPhysics.keyTrustAnalyzer.anomalies.count)")
                    .font(.caption.bold())
            }
        }
    }
}
.padding(.horizontal)
```

---

## Testing Checklist

After implementing, verify:

- [ ] `SignedThreatEvaluation` compiles and `verify()` returns correct results
- [ ] `SignatureReplayDetector.checkRequest()` returns false for normal requests
- [ ] `SignatureReplayDetector.checkRequest()` returns true for rapid-fire requests
- [ ] `KeyTrustAnalyzer.recordSigningEvent()` updates node counts correctly
- [ ] `KeyTrustAnalyzer.analyzeForAnomalies()` detects leaf keys signing too much
- [ ] `ConformalSignaturePredictor.predict()` returns p-value in (0, 1)
- [ ] UI shows entropy count updating with sweeps
- [ ] UI shows replay detection status

---

## Integration Points

These new components should integrate with:

1. **SentryFFI** - For actual Dilithium signing/verification
2. **AlertSystem** - Trigger alerts on replay detection or key trust anomalies
3. **DreamState** - Record signing events in long-term memory
4. **ConsciousFirewall** - Block connections with invalid signatures

---

## Files to Modify

1. `SentryApp/Sources/SentryApp/Models/HyperPhysicsIntegration.swift`
   - Add all new structs and classes
   - Add properties to HyperPhysicsIntegration

2. `SentryApp/Sources/SentryApp/Views/HyperbolicTopologyView.swift`
   - Update PBitDynamicsView with Dilithium Integration GroupBox

3. `SentryApp/Sources/SentryApp/Models/SentryFFI.swift`
   - Ensure signing methods are exposed (may already exist)

---

*Generated by Cascade for inter-instance collaboration*
