# Mathematical Validation Suite - Wolfram Language

Comprehensive validation for both HyperPhysics (9 phases) and Dilithium MCP (7 mathematical domains) using Wolfram Language symbolic computation and numerical analysis.

## Overview

This validation suite provides formal mathematical verification for:
1. **HyperPhysics Foundation** - 9 phases of mathematical foundations
2. **Dilithium MCP** - 7 mathematical domains for the native MCP server

Each validation file serves as the ground truth for correctness verification, ensuring implementations match theoretical formulations.

## Suite 1: Dilithium MCP Validation

**File**: `wolfram-validation-suite.wl`

This comprehensive suite validates all mathematical operations in the dilithium-mcp native Rust implementation.

### Section 1: Hyperbolic Geometry (H¹¹)

**Validated Operations**:
- `lorentz_inner(x, y)` → ⟨x,y⟩_L = -x₀y₀ + Σᵢxᵢyᵢ
- `hyperbolic_distance(p, q)` → d(p,q) = acosh(-⟨p,q⟩_L)
- `lift_to_hyperboloid(z)` → x₀ = √(1 + ||z||²)
- `mobius_add(x, y, c)` → Möbius addition in Poincaré ball

**Properties Verified**:
- Lorentz signature: (-,+,+,...,+)
- Triangle inequality: d(p,q) + d(q,r) ≥ d(p,r)
- Symmetry: d(p,q) = d(q,p)
- Möbius group: identity, inverse, closure

### Section 2: pBit/Ising Statistical Physics

**Validated Operations**:
- `ising_critical_temp()` → T_c = 2/ln(1+√2) ≈ 2.269185314213022
- `boltzmann_weight(E, T)` → exp(-E/T)
- `pbit_probability(h, b, T)` → P(s=1) = 1/(1 + exp(-(h-b)/T))

**Properties Verified**:
- Onsager's exact solution for 2D Ising model
- Boltzmann statistics across temperature regimes
- Ferromagnetic/antiferromagnetic alignment

### Section 3: STDP Learning

**Validated Operations**:
- `stdp_weight_change(Δt, A+, A-, τ)`
  - LTP: ΔW = A+ × exp(-Δt/τ) for Δt > 0
  - LTD: ΔW = -A- × exp(Δt/τ) for Δt < 0

**Properties Verified**:
- Temporal asymmetry (causality)
- Discontinuity at Δt = 0
- Biological realism: A- > A+ (LTD stronger than LTP)

### Section 4: Free Energy Principle

**Validated Operations**:
- `agency_compute_free_energy(obs, beliefs, precision)`
  - F = D_KL[q||p] + E_q[-ln p(o|m)]

**Properties Verified**:
- Non-negativity: F ≥ 0
- Prediction error contribution
- Complexity-accuracy tradeoff

### Section 5: Integrated Information Theory (IIT Φ)

**Validated Operations**:
- `agency_compute_phi(network_state)`
  - Φ = min_partition EI(X; M(X))

**Properties Verified**:
- Non-negativity: Φ ≥ 0
- Boundedness: Φ ≤ 10 (implementation limit)
- Consciousness threshold: Φ > 1.0

### Section 6: Systems Dynamics

**Validated Operations**:
- RK4 integration: `systems_model_simulate(...)`
- Newton-Raphson: `systems_equilibrium_find(...)`

**Properties Verified**:
- RK4 accuracy: < 0.001 error vs analytical solutions
- Newton-Raphson convergence: < 10⁻⁸ tolerance
- Equilibrium stability classification

### Section 7: Self-Organized Criticality

**Validated Operations**:
- `agency_analyze_criticality(timeseries)`
  - Branching ratio σ
  - Hurst exponent H

**Properties Verified**:
- Critical state: σ ≈ 1.0
- Long-range correlations: 0.5 < H < 1
- Power-law avalanche distributions

**Run Dilithium MCP Validation**:
```bash
wolframscript -file wolfram-validation-suite.wl
```

---

## Suite 2: HyperPhysics Validation Phases

### Phase 1: Hyperbolic Geometry
**File**: `phase1_hyperbolic_geometry.wl`

Validates:
- Lorentz inner product: `⟨x,y⟩_L = -x₀y₀ + Σᵢxᵢyᵢ`
- Hyperbolic distance: `d(p,q) = acosh(-⟨p,q⟩_L)`
- Möbius addition in Poincaré ball
- Exponential and logarithmic maps

**Key Tests**:
- Signature verification (timelike, spacelike, lightlike)
- Triangle inequality
- Exp-Log inverse property
- Geodesic tracing

### Phase 2: Learning Algorithms
**File**: `phase2_learning.wl`

Validates:
- Eligibility traces: `e(t) = λγe(t-1) + ∇w`
- STDP learning rule: `ΔW = A₊exp(-Δt/τ₊)` or `-A₋exp(Δt/τ₋)`
- TD(λ) convergence bounds
- Robbins-Monro conditions

**Key Tests**:
- Trace accumulation and decay
- Hebbian consistency
- Temporal credit assignment
- Convergence rate analysis

### Phase 3: Neural Networks
**File**: `phase3_networks.wl`

Validates:
- LIF dynamics: `V(t+1) = leak·V(t) + (1-leak)·I(t)`
- CLIF surrogate gradient
- Watts-Strogatz small-world networks
- Clustering coefficients

**Key Tests**:
- Subthreshold dynamics
- Spike generation
- f-I curves
- Network phase transitions

### Phase 4: Curvature and Graph Geometry
**File**: `phase4_curvature.wl`

Validates:
- Forman-Ricci curvature: `κ(e) = w(e)(deg(v) + deg(w)) - Σ√(w(e)w(e'))`
- Ollivier-Ricci curvature
- HNSW layer probability: `P(layer) = 1 - exp(-1/mL)`
- Sectional curvature

**Key Tests**:
- Positive curvature (complete graphs)
- Negative curvature (trees)
- Hierarchical structure validation
- Curvature-topology relationships

### Phase 5: Adaptive Curvature
**File**: `phase5_adaptive_curvature.wl`

Validates:
- Dynamic curvature: `κ(x,t) = -1/(1 + σ·InformationDensity[x,t])`
- Geodesic attention weights
- Curvature adaptation dynamics

**Key Tests**:
- Spatial and temporal curvature evolution
- Attention mechanism validation
- Gradient-based learning
- Multi-objective optimization

### Phase 6: Autopoiesis and Consciousness
**File**: `phase6_autopoiesis.wl`

Validates:
- Ising critical temperature: `T_c = 2/ln(1+√2)`
- IIT Phi calculation
- Self-organized criticality (BTW sandpile)
- Power law distributions

**Key Tests**:
- Phase transitions
- Integrated information
- Avalanche dynamics
- Critical exponents

### Phase 7: Temporal Dynamics
**File**: `phase7_temporal.wl`

Validates:
- Hyperbolic time embedding: `(sinh(ln(1+t)), cosh(ln(1+t)))`
- Free energy principle: `F = KL(q||p) - H(q)`
- Temporal coherence measures
- Predictive processing

**Key Tests**:
- Logarithmic time scaling
- Free energy minimization
- Autocorrelation decay
- Active inference

### Phase 8: Morphogenetic Fields
**File**: `phase8_morphogenetic.wl`

Validates:
- Heat kernel on H^n: `K(d,t) = (4πt)^(-n/2) exp(-d²/4t)`
- Reaction-diffusion equations (Gray-Scott)
- Turing instability conditions
- Morphogen gradients

**Key Tests**:
- Heat diffusion
- Pattern formation
- French flag model
- Wavelength selection

### Phase 9: Holonomic Processing
**File**: `phase9_holonomic.wl`

Validates:
- Wave interference: `ψ = Σ Aᵢ exp(iφᵢ)`
- Gerchberg-Saxton phase retrieval
- Holographic memory
- Complex amplitude reconstruction

**Key Tests**:
- Constructive/destructive interference
- Double-slit patterns
- Phase recovery algorithms
- Associative recall

## Running Validations

### Option 1: Run Dilithium MCP Validation (Quick)

```bash
cd /Volumes/Tengritek/Ashina/HyperPhysics/tools/dilithium-mcp/validation
wolframscript -file wolfram-validation-suite.wl
```

Validates all dilithium-mcp mathematical operations in ~10 seconds.

### Option 2: Run All HyperPhysics Phases (Comprehensive)

```bash
wolframscript -file run_all_validations.wl
```

This executes all 9 HyperPhysics validation phases sequentially and generates a comprehensive report with:
- Pass/fail status for each phase
- Execution times
- Mathematical coverage summary
- Performance metrics
- Exported JSON results

### Option 3: Run Individual Phases

```bash
# Run a specific phase
wolframscript -file phase1_hyperbolic_geometry.wl

# Run dilithium MCP suite
wolframscript -file wolfram-validation-suite.wl

# Run multiple phases
wolframscript -file phase2_learning.wl
wolframscript -file phase3_networks.wl
```

### Option 4: Interactive Wolfram Notebook

1. Open Mathematica/Wolfram Desktop
2. Load a validation file: `Get["wolfram-validation-suite.wl"]`
3. Results display interactively with formatted output

## Integration with Rust Development

### Workflow

1. **Mathematical Specification** → Define formulas in Wolfram Language
2. **Validation** → Run comprehensive tests to verify correctness
3. **Rust Implementation** → Translate validated formulas to Rust
4. **Cross-Validation** → Compare Rust outputs with Wolfram results
5. **Runtime Verification** → Use Wolfram bridge for critical calculations

### Example Bridge Usage

```rust
use hyperphysics_wolfram::WolframBridge;

let bridge = WolframBridge::new()?;

// Validate hyperbolic distance computation
let rust_distance = hyperbolic_distance(p1, p2);
let wolfram_distance = bridge.validate_expression(
    format!("HyperbolicDistance[{:?}, {:?}]", p1, p2)
)?;

assert!((rust_distance - wolfram_distance).abs() < 1e-10);
```

## Output Format

### Console Output
Each validation file prints:
- Test descriptions
- Numerical results
- Analytical comparisons
- Error bounds
- Pass/fail assertions

### Master Validation Report
`run_all_validations.wl` generates:
- Summary table of all phases
- Execution time statistics
- Mathematical coverage matrix
- Exported JSON results file

### Example Output

```
================================================================================
PHASE 1: HYPERBOLIC GEOMETRY - COMPREHENSIVE VALIDATION
================================================================================

=== LORENTZ INNER PRODUCT VALIDATION ===

Test 1 - Basic: ⟨{2,1,0,0},{3,0,1,0}⟩_L = -6
Test 2 - Symbolic: -s t+x1 y1+x2 y2+x3 y3
Signature: (-,+,+,+)

✓ ALL LORENTZ INNER PRODUCT TESTS PASSED
```

## Requirements

### Software
- **WolframScript**: Version 12.0 or higher
- **Wolfram Desktop/Mathematica**: For interactive use (optional)
- **WolframScript.app**: For HyperPhysics integration (Pro subscription)

### System
- macOS, Linux, or Windows
- Minimum 4GB RAM (8GB recommended for parallel execution)
- 500MB disk space for validation suite

## Error Handling

All validation files use `Assert[]` for critical tests:
- Assertions halt execution on failure
- Error messages indicate which test failed
- Check logs for numerical precision issues

### Common Issues

**Numerical Precision**
```wolfram
(* Adjust tolerance for floating-point comparisons *)
Assert[Abs[computed - expected] < 10^-10]
```

**Memory Limits**
```wolfram
(* Reduce grid sizes for large computations *)
gridSize = 50  (* instead of 100 *)
```

**Performance**
```wolfram
(* Use parallel computation for intensive tests *)
ParallelTable[...] instead of Table[...]
```

## Contributing

To add new validations:

1. Create new validation file: `phaseN_component.wl`
2. Follow existing structure:
   - BeginPackage declaration
   - Function definitions with `::usage` strings
   - Validate* functions for each component
   - Comprehensive test coverage
   - Clear console output
   - EndPackage
3. Add to `run_all_validations.wl`
4. Update this README

## Mathematical Tolerances

```wolfram
$NumericalTolerance = 10^-10       (* Symbolic computation *)
$MachinePrecisionTolerance = 10^-8  (* Machine precision *)
```

These account for:
- Floating-point representation errors
- Numerical stability in edge cases
- Rounding in iterative algorithms

## Mathematical Foundations

All formulas are sourced from peer-reviewed literature:

### Dilithium MCP References
- **Hyperbolic Geometry**: Cannon et al. (1997). "Hyperbolic Geometry"
- **Ising Model**: Onsager (1944). "Crystal Statistics"
- **STDP**: Bi & Poo (1998). "Synaptic Modifications in Cultured Hippocampal Neurons"
- **Free Energy Principle**: Friston (2010). "The free-energy principle: a unified brain theory?"
- **IIT**: Tononi et al. (2016). "Integrated information theory"
- **Self-Organized Criticality**: Bak et al. (1987). "Self-organized criticality"

### HyperPhysics References
- **Hyperbolic Geometry**: Cannon et al. (1997), Anderson (2005)
- **STDP**: Bi & Poo (1998), Song et al. (2000)
- **Ricci Curvature**: Forman (2003), Ollivier (2009)
- **IIT**: Tononi et al. (2016), Oizumi et al. (2014)
- **Free Energy**: Friston (2010), Parr et al. (2022)
- **Morphogenesis**: Turing (1952), Gierer & Meinhardt (1972)

## License

Validation code is provided under the same license as HyperPhysics (see project root LICENSE file).

## Support

For validation issues:
1. Check console output for specific test failures
2. Review mathematical formulas against cited papers
3. Verify WolframScript installation
4. Open GitHub issue with validation logs

---

**Last Updated**: 2025-12-10
**Version**: 2.0.0
**HyperPhysics & Dilithium MCP**: Comprehensive Mathematical Validation Suite
