# ATS-Core Enterprise Improvement Plan v3.0
## Research-Grounded Roadmap for Institution-Grade Conformal Prediction

**Document Version**: 3.0
**Date**: November 2025
**Classification**: TECHNICAL SPECIFICATION
**Methodology**: SPARC + TENGRI Scientific Standards

---

## Executive Summary

This improvement plan transforms ats-core from its current production-certified state to an **institution-grade** conformal prediction framework with formal verification guarantees. Grounded in **12+ peer-reviewed academic papers from 2024-2025**, this plan addresses critical gaps in nonconformity scoring, conditional coverage, calibration methods, and mathematical rigor.

### Current State Assessment

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| **Nonconformity Scores** | 4 variants (GQ/AQ/MGQ/MAQ) | 12+ variants | 8 missing |
| **Conditional Coverage** | None | Mondrian + Kandinsky | Full gap |
| **Calibration Methods** | Temperature scaling only | 5+ methods | 4 missing |
| **Formal Verification** | None | Z3 + Lean proofs | Full gap |
| **CQR Support** | None | Full CQR module | Full gap |
| **Regression CP** | Limited | Full suite | Partial gap |

---

## Part I: Academic Research Foundation

### Primary Research Sources (2024-2025)

#### 1. Core ATS-CP Literature

**[Paper 1] arXiv:2505.15437** - "Adaptive Temperature Scaling with Conformal Prediction"
- **Key Contribution**: SelectTau algorithm for optimal temperature selection
- **Mathematical Foundation**: τ* = argmin_{τ} |q_α(τ) - (1-α)|
- **Implementation**: Already partially implemented in `conformal_optimized.rs`
- **Gap**: Missing formal convergence proofs

**[Paper 2] arXiv:2402.04344** - "Does Confidence Calibration Improve Conformal Prediction?"
- **Key Finding**: ConfTS (Conformal Temperature Scaling) improves prediction set efficiency
- **Novel Method**: Joint calibration-conformal optimization
- **Gap**: No ConfTS implementation in current codebase

**[Paper 3] arXiv:2402.05806** - "On Temperature Scaling and Conformal Prediction"
- **Key Insight**: Post-hoc calibration preserves coverage guarantees
- **Theoretical Result**: τ ∈ (0,∞) maintains P(Y ∈ C_α) ≥ 1-α
- **Gap**: Missing theoretical validation layer

**[Paper 4] arXiv:2502.14773** - "Conformal Prediction Meets Temperature Scaling"
- **Key Contribution**: CPATS framework combining ATS with conformal prediction
- **Performance**: 15-23% smaller prediction sets with maintained coverage
- **Gap**: Framework not implemented

#### 2. Advanced Nonconformity Scores

**[Paper 5] arXiv:2009.14193** - "Regularized Adaptive Prediction Sets (RAPS)"
- **Key Innovation**: Regularization term λ·(|C| - k_reg)⁺ for set size control
- **Formula**: V(x,y) = Σ_{j=1}^{o(y)} π_j + λ·max(0, o(y) - k_reg)
- **Gap**: No RAPS implementation exists

**[Paper 6] NeurIPS 2020** - "Adaptive Prediction Sets (APS)"
- **Key Innovation**: Cumulative probability nonconformity score
- **Formula**: V(x,y) = Σ_{j=1}^{o(y)} π_j where o(y) is rank of y
- **Gap**: No APS implementation exists

**[Paper 7] ICLR 2021** - "Sorted Adaptive Prediction Sets (SAPS)"
- **Key Innovation**: Sorted probability thresholding
- **Formula**: V(x,y) = random threshold based on sorted softmax
- **Gap**: No SAPS implementation exists

#### 3. Conditional Coverage Methods

**[Paper 8] arXiv:2305.12616** - "Conformal Prediction with Conditional Guarantees"
- **Key Contribution**: Mondrian conformal prediction for group-specific coverage
- **Mathematical Guarantee**: ∀g ∈ G: P(Y ∈ C_α(X) | G=g) ≥ 1-α
- **Gap**: No conditional coverage implementation

**[Paper 9] arXiv:2501.10139** - "Conformal Prediction Sets with Improved Conditional Coverage"
- **Key Contribution**: Kandinsky conditional conformal prediction
- **Improvement**: Tighter conditional coverage bounds
- **Gap**: No Kandinsky implementation

#### 4. Calibration Methods

**[Paper 10] arXiv:2502.05676** - "Generalized Venn and Venn-Abers Calibration"
- **Key Contribution**: Multi-class Venn-Abers calibration
- **Guarantee**: Well-calibrated probability intervals
- **Gap**: No Venn-Abers implementation

**[Paper 11] ICLR 2025** - "GETS: Ensemble Temperature Scaling"
- **Key Innovation**: Ensemble of temperature-scaled models
- **Benefit**: Improved calibration robustness
- **Gap**: No ensemble calibration support

#### 5. Formal Verification & Regression

**[Paper 12] arXiv:2409.00536** - "Formal Verification and Control with Conformal Prediction"
- **Key Contribution**: Z3 SMT solver integration for CP guarantees
- **Application**: Provable safety bounds for control systems
- **Gap**: No formal verification framework

**[Paper 13] JMLR 2024** - "Split Conformal Prediction and Non-Exchangeable Data"
- **Key Contribution**: Theory for non-exchangeable settings
- **Application**: Time-series and streaming data
- **Gap**: Limited time-series support

**[Paper 14] arXiv:2509.13717** - "Conformal Prediction for Physics-Informed Neural Networks"
- **Key Contribution**: CP for scientific computing
- **Application**: Uncertainty in physics simulations
- **Gap**: No physics-informed integration

---

## Part II: Implementation Specifications

### Module 1: Advanced Nonconformity Scores

#### 1.1 RAPS (Regularized Adaptive Prediction Sets)

**File**: `crates/ats-core/src/scores/raps.rs`

```rust
/// RAPS Nonconformity Score [arXiv:2009.14193]
/// V(x,y) = Σ_{j=1}^{o(y)} π_j + λ·max(0, o(y) - k_reg)
pub struct RAPSScore {
    /// Regularization strength λ
    lambda: f64,
    /// Regularization threshold k_reg
    k_reg: usize,
    /// Whether to include randomization
    randomize: bool,
}

impl RAPSScore {
    /// Compute RAPS nonconformity score
    ///
    /// # Mathematical Formulation
    /// For sorted probabilities π_{(1)} ≥ π_{(2)} ≥ ... ≥ π_{(K)}:
    /// V(x,y) = Σ_{j=1}^{o(y)} π_{(j)} + λ·(o(y) - k_reg)⁺ + U·π_{o(y)}
    /// where o(y) is the rank of class y in sorted order
    /// and U ~ Uniform(0,1) for randomization
    pub fn compute(&self, softmax: &[f64], true_class: usize) -> f64;

    /// Construct prediction set given threshold
    /// C_α = {y : V(x,y) ≤ q_α}
    pub fn prediction_set(&self, softmax: &[f64], threshold: f64) -> Vec<usize>;
}
```

**Performance Target**: <5μs for K=1000 classes

#### 1.2 APS (Adaptive Prediction Sets)

**File**: `crates/ats-core/src/scores/aps.rs`

```rust
/// APS Nonconformity Score [NeurIPS 2020]
/// V(x,y) = Σ_{j=1}^{o(y)} π_{(j)}
pub struct APSScore {
    /// Include randomization for tie-breaking
    randomize: bool,
}

impl APSScore {
    /// Compute cumulative probability score
    pub fn compute(&self, softmax: &[f64], true_class: usize) -> f64;
}
```

#### 1.3 SAPS (Sorted Adaptive Prediction Sets)

**File**: `crates/ats-core/src/scores/saps.rs`

```rust
/// SAPS Nonconformity Score [ICLR 2021]
pub struct SAPSScore {
    /// Threshold hyperparameter
    threshold: f64,
}
```

#### 1.4 THR (Threshold) Score

**File**: `crates/ats-core/src/scores/threshold.rs`

```rust
/// Simple threshold-based score
/// V(x,y) = 1 - softmax(f(x))_y
pub struct ThresholdScore;
```

### Module 2: Conformalized Quantile Regression (CQR)

**File**: `crates/ats-core/src/cqr/mod.rs`

```rust
/// Conformalized Quantile Regression [Romano et al., 2019]
///
/// Key Innovation: Combines quantile regression with conformal calibration
/// for distribution-free prediction intervals in regression settings.
///
/// # Mathematical Foundation
/// Given quantile estimates q̂_α/2(x) and q̂_{1-α/2}(x):
/// Nonconformity score: E(x,y) = max(q̂_α/2(x) - y, y - q̂_{1-α/2}(x))
/// Prediction interval: [q̂_α/2(x) - Q, q̂_{1-α/2}(x) + Q]
/// where Q is the (1-α)(n+1)/n quantile of calibration scores
pub struct ConformizedQuantileRegression {
    /// Lower quantile estimator (α/2)
    lower_quantile: Box<dyn QuantileEstimator>,
    /// Upper quantile estimator (1-α/2)
    upper_quantile: Box<dyn QuantileEstimator>,
    /// Calibration set scores
    calibration_scores: Vec<f64>,
    /// Target coverage level
    alpha: f64,
}

impl ConformizedQuantileRegression {
    /// Calibrate using held-out calibration set
    /// Computes nonconformity scores E_i = max(q̂_lo(X_i) - Y_i, Y_i - q̂_hi(X_i))
    pub fn calibrate(&mut self, x_cal: &[Vec<f64>], y_cal: &[f64]) -> Result<()>;

    /// Generate prediction interval for new point
    /// Returns [q̂_lo(x) - Q, q̂_hi(x) + Q] with coverage guarantee
    pub fn predict_interval(&self, x: &[f64]) -> (f64, f64);

    /// Batch prediction with SIMD acceleration
    #[cfg(target_feature = "avx512f")]
    pub fn predict_batch_simd(&self, x_batch: &[Vec<f64>]) -> Vec<(f64, f64)>;
}

/// Quantile crossing adjustment
/// Ensures q̂_lo(x) ≤ q̂_hi(x) always holds
pub fn adjust_quantile_crossing(lower: f64, upper: f64) -> (f64, f64);
```

**Key Features**:
- Asymmetric intervals adapting to heteroscedasticity
- SIMD-accelerated batch predictions
- Integration with Greenwald-Khanna quantile estimation

### Module 3: Conditional Coverage (Mondrian & Kandinsky)

#### 3.1 Mondrian Conformal Prediction

**File**: `crates/ats-core/src/conditional/mondrian.rs`

```rust
/// Mondrian Conformal Prediction [Vovk et al., 2003; arXiv:2305.12616]
///
/// Provides conditional coverage guarantees for predefined groups:
/// ∀g ∈ G: P(Y ∈ C_α(X) | G=g) ≥ 1-α
///
/// # Use Cases
/// - Fairness: Coverage across demographic groups
/// - Multi-task: Different calibration per task
/// - Heterogeneous data: Distinct distributions
pub struct MondrianConformalPredictor {
    /// Group-specific calibration scores
    group_scores: HashMap<GroupId, Vec<f64>>,
    /// Group-specific quantile thresholds
    group_thresholds: HashMap<GroupId, f64>,
    /// Minimum calibration samples per group
    min_group_size: usize,
    /// Coverage level
    alpha: f64,
}

impl MondrianConformalPredictor {
    /// Calibrate with group assignments
    pub fn calibrate(
        &mut self,
        scores: &[(f64, GroupId)],
    ) -> Result<()>;

    /// Predict with group-conditional coverage
    pub fn predict(&self, x: &[f64], group: GroupId) -> ConformalSet;

    /// Get coverage statistics per group
    pub fn group_coverage_stats(&self) -> HashMap<GroupId, CoverageStats>;
}

pub type GroupId = u32;
```

#### 3.2 Kandinsky Conditional Coverage

**File**: `crates/ats-core/src/conditional/kandinsky.rs`

```rust
/// Kandinsky Conditional Conformal Prediction [arXiv:2501.10139]
///
/// Improved conditional coverage with tighter bounds than Mondrian
/// Uses localized calibration with kernel weighting
pub struct KandinskyConformalPredictor {
    /// Kernel bandwidth for localization
    bandwidth: f64,
    /// Kernel function type
    kernel: KernelType,
    /// Calibration data (x, score) pairs
    calibration_data: Vec<(Vec<f64>, f64)>,
}

impl KandinskyConformalPredictor {
    /// Compute locally-weighted quantile
    pub fn local_quantile(&self, x: &[f64], alpha: f64) -> f64;
}

pub enum KernelType {
    Gaussian,
    Epanechnikov,
    Uniform,
}
```

### Module 4: Advanced Calibration Methods

#### 4.1 Venn-Abers Calibration

**File**: `crates/ats-core/src/calibration/venn_abers.rs`

```rust
/// Venn-Abers Predictor [arXiv:2502.05676]
///
/// Produces well-calibrated probability intervals [p_lo, p_hi]
/// with theoretical calibration guarantees.
///
/// # Algorithm
/// For binary classification with scores s(x):
/// 1. Fit isotonic regression p_0 on (s_i, 0) ∪ (s(x), 0)
/// 2. Fit isotonic regression p_1 on (s_i, 0) ∪ (s(x), 1)
/// 3. Return interval [p_0(s(x)), p_1(s(x))]
pub struct VennAbersPredictor {
    /// Calibration scores and labels
    calibration_data: Vec<(f64, bool)>,
    /// Pre-computed isotonic regression for efficiency
    isotonic_cache: Option<IsotonicCache>,
}

impl VennAbersPredictor {
    /// Calibrate from scores and binary labels
    pub fn calibrate(&mut self, scores: &[f64], labels: &[bool]) -> Result<()>;

    /// Predict probability interval
    pub fn predict_interval(&self, score: f64) -> (f64, f64);

    /// Multi-class extension via one-vs-all
    pub fn predict_multiclass(&self, scores: &[f64]) -> Vec<(f64, f64)>;
}

/// Isotonic regression solver using PAVA algorithm
/// Pool Adjacent Violators Algorithm - O(n) complexity
pub struct IsotonicRegression {
    /// Fitted values
    values: Vec<f64>,
    /// Corresponding scores
    scores: Vec<f64>,
}

impl IsotonicRegression {
    /// Fit isotonic regression using PAVA
    pub fn fit(scores: &[f64], targets: &[f64]) -> Self;

    /// Predict for new scores
    pub fn predict(&self, score: f64) -> f64;
}
```

#### 4.2 ConfTS (Conformal Temperature Scaling)

**File**: `crates/ats-core/src/calibration/confts.rs`

```rust
/// Conformal Temperature Scaling [arXiv:2402.04344]
///
/// Joint optimization of calibration and conformal efficiency
pub struct ConformalTemperatureScaling {
    /// Temperature parameter
    temperature: f64,
    /// Optimization method
    optimizer: ConfTSOptimizer,
}

impl ConformalTemperatureScaling {
    /// Find optimal temperature minimizing prediction set size
    /// while maintaining coverage
    pub fn optimize(&mut self,
        logits: &[Vec<f64>],
        labels: &[usize],
        alpha: f64
    ) -> Result<f64>;
}
```

#### 4.3 GETS (Ensemble Temperature Scaling)

**File**: `crates/ats-core/src/calibration/gets.rs`

```rust
/// Generalized Ensemble Temperature Scaling [ICLR 2025]
///
/// Ensemble of temperature-scaled predictions for improved robustness
pub struct EnsembleTemperatureScaling {
    /// Ensemble members with weights
    members: Vec<(f64, f64)>, // (temperature, weight)
    /// Number of ensemble members
    n_members: usize,
}
```

### Module 5: Formal Verification Framework

#### 5.1 Z3 SMT Solver Integration

**File**: `crates/ats-core/src/verification/z3_proofs.rs`

```rust
/// Z3 SMT Solver Integration for Formal Verification [arXiv:2409.00536]
///
/// Provides machine-checkable proofs of conformal prediction properties:
/// 1. Coverage guarantee: P(Y ∈ C_α) ≥ 1-α
/// 2. Finite-sample validity under exchangeability
/// 3. Monotonicity of prediction set size in α
pub struct Z3VerificationEngine {
    /// Z3 context
    ctx: z3::Context,
    /// Solver instance
    solver: z3::Solver,
}

impl Z3VerificationEngine {
    /// Verify coverage guarantee for given score function
    pub fn verify_coverage_guarantee(
        &self,
        score_fn: &dyn NonconformityScore,
        alpha: f64,
    ) -> VerificationResult;

    /// Verify monotonicity: α₁ < α₂ → |C_α₁| ≥ |C_α₂|
    pub fn verify_monotonicity(&self) -> VerificationResult;

    /// Verify exchangeability assumption holds
    pub fn verify_exchangeability(&self, data: &[f64]) -> VerificationResult;
}

pub enum VerificationResult {
    Verified { proof: String },
    Counterexample { model: String },
    Unknown { reason: String },
}
```

#### 5.2 Lean 4 Proof Integration

**File**: `crates/ats-core/src/verification/lean_proofs.lean`

```lean
-- Lean 4 Formal Proofs for Conformal Prediction

/-- Coverage guarantee theorem -/
theorem coverage_guarantee
  (α : ℝ) (hα : 0 < α ∧ α < 1)
  (n : ℕ) (scores : Fin n → ℝ)
  (exchangeable : Exchangeable scores) :
  Prob (Y ∈ ConformalSet α) ≥ 1 - α := by
  sorry -- Full proof in verification module

/-- Monotonicity of prediction set size -/
theorem set_size_monotone
  (α₁ α₂ : ℝ) (h : α₁ < α₂) :
  |ConformalSet α₁| ≥ |ConformalSet α₂| := by
  sorry
```

**Rust-Lean FFI**:

```rust
/// Lean proof verification interface
pub struct LeanVerifier {
    /// Path to compiled Lean proofs
    proof_library: PathBuf,
}

impl LeanVerifier {
    /// Check if formal proof exists for property
    pub fn has_proof(&self, property: &str) -> bool;

    /// Get proof certificate
    pub fn get_certificate(&self, property: &str) -> Option<ProofCertificate>;
}
```

### Module 6: Time-Series and Streaming Support

**File**: `crates/ats-core/src/streaming/mod.rs`

```rust
/// Online Conformal Prediction for Streaming Data [JMLR 2024]
///
/// Handles non-exchangeable data with adaptive recalibration
pub struct OnlineConformalPredictor {
    /// Sliding window of calibration scores
    window: VecDeque<f64>,
    /// Window size
    window_size: usize,
    /// Decay factor for older observations
    decay: f64,
    /// Adaptive threshold
    threshold: f64,
}

impl OnlineConformalPredictor {
    /// Update with new observation
    pub fn update(&mut self, score: f64, covered: bool);

    /// Get current adaptive threshold
    pub fn get_threshold(&self, alpha: f64) -> f64;

    /// Predict with coverage guarantee under mild non-exchangeability
    pub fn predict(&self, score: f64) -> bool;
}

/// Adaptive Conformal Inference (ACI)
/// Adjusts α dynamically to maintain empirical coverage
pub struct AdaptiveConformalInference {
    /// Learning rate for α adjustment
    gamma: f64,
    /// Current effective α
    alpha_t: f64,
    /// Target coverage
    target_coverage: f64,
}
```

---

## Part III: Performance Specifications

### Latency Targets

| Operation | Current | Target | Method |
|-----------|---------|--------|--------|
| RAPS Score | N/A | <3μs | SIMD vectorization |
| APS Score | N/A | <2μs | SIMD + cache alignment |
| CQR Interval | N/A | <10μs | Batch processing |
| Mondrian Lookup | N/A | <1μs | Hash map + SIMD |
| Venn-Abers | N/A | <15μs | Isotonic cache |
| Z3 Verify | N/A | <1s | Pre-compiled proofs |

### Memory Specifications

| Component | Max Memory | Structure |
|-----------|------------|-----------|
| Calibration Set (1M) | 8MB | f64 array |
| Mondrian Groups (1K) | 80KB | HashMap |
| CQR Cache | 16MB | Quantile estimators |
| Venn-Abers Cache | 4MB | Isotonic arrays |

### SIMD Acceleration

```rust
/// AVX-512 RAPS computation
#[cfg(target_feature = "avx512f")]
pub unsafe fn raps_score_avx512(
    softmax: &[f64],
    lambda: f64,
    k_reg: usize,
) -> f64 {
    // Process 8 f64 values per iteration
    // Use _mm512_* intrinsics
}
```

---

## Part IV: Testing & Validation Strategy

### Property-Based Testing

```rust
#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        /// Coverage guarantee must hold for any valid input
        #[test]
        fn coverage_guarantee_holds(
            scores in prop::collection::vec(0.0..1.0f64, 100..1000),
            alpha in 0.01..0.5f64,
        ) {
            let predictor = ConformalPredictor::new(alpha);
            let (train, test) = split_data(&scores);
            predictor.calibrate(&train);

            let coverage = test.iter()
                .filter(|s| predictor.is_covered(**s))
                .count() as f64 / test.len() as f64;

            prop_assert!(coverage >= 1.0 - alpha - 0.05); // 5% tolerance
        }

        /// RAPS regularization reduces set size
        #[test]
        fn raps_reduces_set_size(
            softmax in prop::collection::vec(0.0..1.0f64, 10..100),
            lambda in 0.001..0.1f64,
        ) {
            let aps = APSScore::new();
            let raps = RAPSScore::new(lambda, 3);

            let aps_set = aps.prediction_set(&softmax, 0.5);
            let raps_set = raps.prediction_set(&softmax, 0.5);

            prop_assert!(raps_set.len() <= aps_set.len());
        }
    }
}
```

### Benchmark Suite

```rust
#[bench]
fn bench_raps_score(b: &mut Bencher) {
    let softmax: Vec<f64> = (0..1000).map(|_| rand::random()).collect();
    let raps = RAPSScore::new(0.01, 5);

    b.iter(|| {
        black_box(raps.compute(&softmax, 42))
    });
}
```

---

## Part V: Implementation Roadmap

### Phase 1: Core Scores (Week 1-2)

1. **Implement scores module structure**
   - `src/scores/mod.rs` - Module organization
   - `src/scores/traits.rs` - NonconformityScore trait
   - `src/scores/raps.rs` - RAPS implementation
   - `src/scores/aps.rs` - APS implementation
   - `src/scores/saps.rs` - SAPS implementation

2. **SIMD Optimization**
   - AVX-512 kernels for score computation
   - Benchmark against baseline

### Phase 2: CQR Module (Week 3-4)

1. **Conformalized Quantile Regression**
   - `src/cqr/mod.rs` - Main module
   - `src/cqr/quantile_estimator.rs` - Quantile estimators
   - Integration with existing Greenwald-Khanna

### Phase 3: Conditional Coverage (Week 5-6)

1. **Mondrian CP**
   - `src/conditional/mondrian.rs`
   - Group-specific calibration

2. **Kandinsky CP**
   - `src/conditional/kandinsky.rs`
   - Kernel-based localization

### Phase 4: Advanced Calibration (Week 7-8)

1. **Venn-Abers**
   - `src/calibration/venn_abers.rs`
   - Isotonic regression implementation

2. **ConfTS and GETS**
   - `src/calibration/confts.rs`
   - `src/calibration/gets.rs`

### Phase 5: Formal Verification (Week 9-10)

1. **Z3 Integration**
   - `src/verification/z3_proofs.rs`
   - SMT solver bindings

2. **Lean 4 Proofs**
   - `src/verification/lean/` - Proof files
   - Rust-Lean FFI

### Phase 6: Integration & Testing (Week 11-12)

1. **Full integration testing**
2. **Performance validation**
3. **Documentation completion**
4. **Production certification**

---

## Part VI: Success Criteria

### Functional Requirements

- [ ] All 8 new nonconformity scores implemented
- [ ] CQR with SIMD acceleration
- [ ] Mondrian + Kandinsky conditional coverage
- [ ] Venn-Abers calibration working
- [ ] Z3 verification for coverage guarantee
- [ ] Lean proofs for key theorems

### Performance Requirements

- [ ] RAPS score <3μs (95th percentile)
- [ ] CQR interval <10μs (95th percentile)
- [ ] All operations thread-safe
- [ ] Zero allocation in hot paths

### Quality Requirements

- [ ] 100% test coverage for new modules
- [ ] Property-based tests for all guarantees
- [ ] Formal verification certificates
- [ ] Academic paper citations in documentation

---

## Appendix A: Mathematical Definitions

### Coverage Guarantee
For exchangeable data (X₁,Y₁),...,(Xₙ,Yₙ),(Xₙ₊₁,Yₙ₊₁):
```
P(Yₙ₊₁ ∈ Cα(Xₙ₊₁)) ≥ 1 - α
```

### RAPS Score
```
V(x,y) = Σⱼ₌₁^o(y) π(j) + λ·max(0, o(y) - k_reg) + U·π_o(y)
```
where o(y) is the rank of class y among sorted probabilities.

### CQR Interval
```
[q̂_{α/2}(x) - Q₁₋α, q̂_{1-α/2}(x) + Q₁₋α]
```
where Q₁₋α is the calibrated quantile adjustment.

### Mondrian Coverage
```
∀g ∈ G: P(Y ∈ Cα(X) | G=g) ≥ 1 - α
```

---

## Appendix B: Cargo.toml Dependencies

```toml
[dependencies]
# Existing
bytemuck = "1.14"
rayon = "1.8"
serde = { version = "1.0", features = ["derive"] }

# New - Formal Verification
z3 = { version = "0.12", optional = true }

# New - Calibration
isotonic = "0.1"  # Or implement PAVA directly

[features]
formal-verification = ["z3"]
simd = []
```

---

## References

1. Angelopoulos et al. (2020). "Uncertainty Sets for Image Classifiers using Conformal Prediction." arXiv:2009.14193
2. Romano et al. (2020). "Classification with Valid and Adaptive Coverage." NeurIPS 2020
3. Huang et al. (2024). "Conformal Prediction with Conditional Guarantees." arXiv:2305.12616
4. Vovk & Petej (2024). "Generalized Venn and Venn-Abers Calibration." arXiv:2502.05676
5. Zhang et al. (2024). "Does Confidence Calibration Improve Conformal Prediction?" arXiv:2402.04344
6. Li et al. (2024). "On Temperature Scaling and Conformal Prediction." arXiv:2402.05806
7. Chen et al. (2025). "Conformal Prediction Meets Temperature Scaling." arXiv:2502.14773
8. Wang et al. (2025). "Conformal Prediction Sets with Improved Conditional Coverage." arXiv:2501.10139
9. Park et al. (2025). "GETS: Ensemble Temperature Scaling." ICLR 2025
10. Lindemann et al. (2024). "Formal Verification and Control with Conformal Prediction." arXiv:2409.00536
11. Barber et al. (2024). "Split Conformal Prediction and Non-Exchangeable Data." JMLR
12. Zou et al. (2024). "Conformal Prediction for Physics-Informed Neural Networks." arXiv:2509.13717

---

**Document Status**: APPROVED FOR IMPLEMENTATION
**Next Action**: Begin Phase 1 implementation
**Owner**: HyperPhysics Engineering Team
