# Dilithium MCP Wolfram Validation - Quick Reference Card

## 🚀 Quick Start

```bash
cd /Volumes/Tengritek/Ashina/HyperPhysics/tools/dilithium-mcp/validation
wolframscript -file wolfram-validation-suite.wl
```

## 📋 Mathematical Operations Reference

### 1. Hyperbolic Geometry (H¹¹)

| Operation | Rust Function | Wolfram Formula | Test Tolerance |
|-----------|---------------|-----------------|----------------|
| Lorentz Inner Product | `lorentz_inner(x, y)` | `⟨x,y⟩_L = -x₀y₀ + Σᵢxᵢyᵢ` | 10⁻¹⁰ |
| Hyperbolic Distance | `hyperbolic_distance(p, q)` | `d = acosh(-⟨p,q⟩_L)` | 10⁻⁸ |
| Lift to Hyperboloid | `lift_to_hyperboloid(z)` | `x₀ = √(1 + ‖z‖²)` | 10⁻⁸ |
| Möbius Addition | `mobius_add(x, y, c)` | `(x⊕y) = numerator/denominator` | 10⁻⁸ |

**Expected Properties:**
- Self-inner product on hyperboloid: ⟨x,x⟩_L = -1
- Triangle inequality: d(p,q) + d(q,r) ≥ d(p,r)
- Symmetry: d(p,q) = d(q,p)

### 2. Statistical Physics

| Operation | Rust Function | Wolfram Formula | Expected Value |
|-----------|---------------|-----------------|----------------|
| Ising Critical Temp | `ising_critical_temp()` | `2/ln(1+√2)` | 2.269185314213022 |
| Boltzmann Weight | `boltzmann_weight(E, T)` | `exp(-E/T)` | varies |
| pBit Probability | `pbit_probability(h, b, T)` | `1/(1 + exp(-(h-b)/T))` | [0,1] |

**Expected Properties:**
- At T = T_c: Phase transition occurs
- At h = 0, b = 0, T = 1: P(s=1) = 0.5
- High field: P(s=1) → 1

### 3. STDP Learning

| Condition | Rust Function | Wolfram Formula | Example (Δt=10ms) |
|-----------|---------------|-----------------|-------------------|
| LTP (Δt > 0) | `stdp_weight_change(10, 0.1, 0.12, 20)` | `0.1 × exp(-10/20)` | 0.0606530660 |
| LTD (Δt < 0) | `stdp_weight_change(-10, 0.1, 0.12, 20)` | `-0.12 × exp(10/20)` | -0.1977946552 |

**Expected Properties:**
- Discontinuity at Δt = 0
- Asymmetry: |LTD| > |LTP| (A- > A+)
- Exponential decay with time constant τ

### 4. Free Energy Principle

| Metric | Rust Function | Wolfram Formula | Range |
|--------|---------------|-----------------|-------|
| Free Energy | `agency_compute_free_energy(o, b, p)` | `F = D_KL[q‖p] + accuracy` | F ≥ 0 |

**Expected Properties:**
- Non-negativity: F ≥ 0
- Perfect prediction: F ≈ complexity term only
- Large error: F increases

### 5. Integrated Information (Φ)

| State | Expected Φ | Interpretation |
|-------|-----------|----------------|
| Zero activity | 0 | No consciousness |
| Uniform activation | ≈1 | Moderate integration |
| Complex patterns | 1-10 | High integration |

**Consciousness Threshold:** Φ > 1.0

### 6. Systems Dynamics

| Method | Accuracy | Convergence |
|--------|----------|-------------|
| RK4 Integration | < 0.001 error | vs analytical |
| Newton-Raphson | < 10⁻⁸ tolerance | vs √2 |

### 7. Criticality

| Metric | Critical Value | Interpretation |
|--------|---------------|----------------|
| Branching Ratio (σ) | ≈ 1.0 | Self-organized criticality |
| Hurst Exponent (H) | > 0.5 | Long-range correlations |

## 🔍 Validation Commands

### Run Full Suite
```bash
wolframscript -file wolfram-validation-suite.wl
```

### Run with Verbose Output
```bash
wolframscript -verbose -file wolfram-validation-suite.wl
```

### Run All HyperPhysics + Dilithium
```bash
wolframscript -file run_all_validations.wl
```

## ✅ Test Status Indicators

| Symbol | Meaning |
|--------|---------|
| ✓ PASSED | Test successful within tolerance |
| ✗ FAILED | Test failed, check logs |
| â | Unicode display (same as ✓) |

## 🎯 Common Test Values

### Hyperbolic Geometry
```wolfram
(* Origin in H¹¹ *)
origin = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

(* Point on H¹¹ *)
point = {Cosh[1], Sinh[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

(* Test distance *)
d = HyperbolicDistance[origin, point]  (* Should be 1.0 *)
```

### STDP
```wolfram
(* Standard parameters *)
A_plus = 0.1
A_minus = 0.12
tau = 20  (* ms *)

(* Test LTP *)
dw = STDPWeightChange[10, 0.1, 0.12, 20]  (* 0.06065 *)
```

### Free Energy
```wolfram
(* Perfect prediction *)
obs = {1, 2, 3}
beliefs = {1, 2, 3}
precision = {1, 1, 1}
F = FreeEnergy[obs, beliefs, precision]  (* Minimal, ≥ 0 *)
```

## 📊 Performance Benchmarks

| Suite | Execution Time | Tests |
|-------|----------------|-------|
| Dilithium MCP | ~10 seconds | 14 tests |
| HyperPhysics Phase 1 | ~5 seconds | 15 tests |
| All 9 Phases | ~60 seconds | 135 tests |

## 🐛 Debugging

### Check Wolfram Version
```bash
wolframscript --version
```

### Test Single Function
```wolfram
(* In Mathematica/Wolfram Desktop *)
<< "wolfram-validation-suite.wl"
TestLorentzInnerProduct[]
```

### Export Results
```wolfram
results = RunAllTests[];
Export["validation_results.json", results, "JSON"]
```

## 📚 Mathematical References

| Domain | Key Paper | Year |
|--------|-----------|------|
| Hyperbolic Geometry | Cannon et al. | 1997 |
| Ising Model | Onsager | 1944 |
| STDP | Bi & Poo | 1998 |
| Free Energy | Friston | 2010 |
| IIT | Tononi et al. | 2016 |
| SOC | Bak et al. | 1987 |

## 🔧 Integration with CI/CD

### Pre-commit Hook
```bash
#!/bin/bash
wolframscript -file validation/wolfram-validation-suite.wl | grep "FAILED"
if [ $? -eq 0 ]; then
    echo "Validation FAILED"
    exit 1
fi
```

### GitHub Actions
```yaml
- name: Validate Mathematics
  run: wolframscript -file wolfram-validation-suite.wl
```

---

**Quick Help:** `wolframscript -help`
**Documentation:** See `README.md` in same directory
**Issues:** Check `/tmp/wolfram-validation.log`
