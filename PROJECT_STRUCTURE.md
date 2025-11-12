# HyperPhysics Project Structure

**Complete file organization and navigation guide**

---

## 📁 Root Directory

```
HyperPhysics/
├── README.md                 # Main project documentation (START HERE)
├── STATUS.md                 # Current project status
├── QUICK_START.md           # One-page quick reference
├── CHECKLIST.md             # Interactive execution checklist
├── PROJECT_STRUCTURE.md     # This file
├── PHASE2_README.md         # Phase 2 detailed guide
├── Cargo.toml               # Workspace configuration
├── Cargo.lock               # Dependency lock file
├── .gitignore               # Git ignore patterns
└── LICENSE                  # Project license
```

---

## 📚 Documentation (`/docs/`)

### Phase 2 Documentation:
```
docs/
├── READY_FOR_PHASE2_EXECUTION.md    # Complete readiness report
├── PHASE2_KICKOFF.md                # 6-week Phase 2 plan (15,000+ words)
├── PHASE2_WEEK3_COMPLETE.md         # Week 3 completion report
├── PHASE2_WEEK3_EXECUTION_GUIDE.md  # Step-by-step execution guide
├── SIMD_IMPLEMENTATION_COMPLETE.md  # SIMD technical documentation
```

### Phase 1 Documentation:
```
docs/
├── PHASE1_COMPLETION_REPORT.md      # Complete Phase 1 report
```

### Architecture Documentation:
```
docs/architecture/
├── MARKET_CRATE.md                  # Market data architecture (3,247 lines)
├── REGIME_DETECTION.md              # Consciousness-based detection (5,823 lines)
├── RISK_CRATE.md                    # Risk management (1,654 lines)
└── [Other architecture docs...]
```

### Performance Data:
```
docs/performance/
├── baselines/                       # Scalar performance baselines
│   └── baseline_TIMESTAMP.txt
├── validation/                      # SIMD validation reports
│   └── simd_validation_TIMESTAMP.txt
└── SIMD_COMPARISON.txt             # Scalar vs SIMD comparison
```

### Integration Patches:
```
docs/patches/
└── ENGINE_SIMD_INTEGRATION.patch   # Engine SIMD integration guide
```

---

## 🔧 Scripts (`/scripts/`)

### Installation & Setup:
```
scripts/
├── phase2_setup.sh                 # Complete Rust + Lean 4 installation
└── verify_simd_integration.sh      # Verify SIMD components ready
```

### Testing & Validation:
```
scripts/
├── validate_system.sh              # 10-point system validation
├── benchmark_baseline.sh           # Establish scalar baselines
└── validate_performance.sh         # Comprehensive performance validation
```

### SIMD Deployment:
```
scripts/
└── apply_simd_integration.sh       # Apply engine integration patch
```

### Development Tools:
```
scripts/
├── run_mutation_tests.sh           # Mutation testing
└── run_fuzz_tests.sh              # Fuzzing security tests
```

---

## 📦 Crates (`/crates/`)

### Main Crates:

```
crates/
├── hyperphysics-core/              # Main engine
│   ├── src/
│   │   ├── lib.rs                 # Library exports
│   │   ├── engine.rs              # Main engine (268 lines)
│   │   ├── config.rs              # Configuration
│   │   ├── metrics.rs             # Metrics collection
│   │   ├── crypto.rs              # Ed25519 identity
│   │   ├── gpu.rs                 # GPU foundation
│   │   └── simd/                  # SIMD optimization (Week 3)
│   │       ├── mod.rs             # Module organization (101 lines)
│   │       ├── math.rs            # Vectorized math (290 lines)
│   │       ├── backend.rs         # CPU detection (246 lines)
│   │       └── engine.rs          # Engine integration (139 lines)
│   ├── Cargo.toml
│   └── benches/                   # Benchmarks
│
├── hyperphysics-geometry/          # Hyperbolic geometry
│   ├── src/
│   │   ├── lib.rs
│   │   ├── hyperbolic.rs          # H³ manifold
│   │   ├── poincare.rs            # Poincaré coordinates
│   │   └── tessellation.rs        # {3,7,2} tessellation
│   └── Cargo.toml
│
├── hyperphysics-pbit/              # Probabilistic dynamics
│   ├── src/
│   │   ├── lib.rs
│   │   ├── pbit.rs                # pBit implementation
│   │   ├── lattice.rs             # pBit lattice
│   │   ├── gillespie.rs           # Gillespie SSA
│   │   └── metropolis.rs          # Metropolis MCMC
│   └── Cargo.toml
│
├── hyperphysics-thermo/            # Thermodynamics
│   ├── src/
│   │   ├── lib.rs
│   │   ├── hamiltonian.rs         # Energy calculations
│   │   ├── entropy.rs             # Entropy calculations
│   │   └── landauer.rs            # Landauer principle
│   └── Cargo.toml
│
├── hyperphysics-consciousness/     # Consciousness metrics
│   ├── src/
│   │   ├── lib.rs
│   │   ├── phi.rs                 # Φ (IIT)
│   │   └── ci.rs                  # CI (Resonance)
│   └── Cargo.toml
│
├── hyperphysics-market/            # Market integration
│   ├── src/
│   │   ├── lib.rs
│   │   ├── alpaca.rs              # Alpaca API
│   │   ├── topology.rs            # Market topology
│   │   └── regime.rs              # Regime detection
│   └── Cargo.toml
│
└── hyperphysics-risk/              # Risk management
    ├── src/
    │   ├── lib.rs
    │   ├── portfolio.rs           # Portfolio risk
    │   └── thermodynamic.rs       # Thermo risk
    └── Cargo.toml
```

---

## 🧪 Tests

### Test Organization:
```
crates/*/tests/                     # Integration tests
crates/*/src/*_test.rs             # Unit tests (inline)
crates/*/benches/                   # Benchmarks
```

### Test Commands:
```bash
# All tests
cargo test --workspace

# With SIMD
cargo test --workspace --features simd

# Specific crate
cargo test -p hyperphysics-core

# SIMD tests only
cargo test --features simd --lib simd
```

---

## 🎯 Build Artifacts

```
target/
├── debug/                         # Debug builds
├── release/                       # Release builds
├── criterion/                     # Benchmark data
│   ├── engine_step/
│   ├── entropy_calculation/
│   └── report/                    # HTML reports
└── doc/                          # Generated documentation
```

---

## 📊 Performance Data

### Generated During Execution:
```
docs/performance/
├── baselines/
│   ├── baseline_20251112_*.txt   # Scalar baselines
│   └── ...
├── validation/
│   ├── simd_validation_*.txt     # Validation reports
│   └── ...
└── SIMD_COMPARISON.txt           # Final comparison
```

---

## 🔄 Backup Files

### Created by Scripts:
```
crates/hyperphysics-core/src/
└── engine.rs.backup_*            # Pre-integration backups
```

---

## 📈 Key Metrics

### Current Counts:
- **Total Files**: 77+ (Phase 1) + 4 SIMD modules
- **Lines of Code**: 87,043 (Phase 1) + 776 SIMD
- **Tests**: 91 (Phase 1) + 15 SIMD = 106 total
- **Documentation**: 15+ comprehensive guides
- **Scripts**: 6 automation scripts

---

## 🗂️ File Types

### Source Code:
- `.rs` - Rust source files
- `.toml` - Cargo configuration

### Documentation:
- `.md` - Markdown documentation
- `.patch` - Integration patches

### Scripts:
- `.sh` - Bash automation scripts
- `.py` - Python helpers (in scripts)

### Data:
- `.txt` - Performance baselines
- `.json` - Configuration data

---

## 🔍 Navigation Guide

### Want to...

**Start execution?**
→ Read `README.md` → Run `./scripts/phase2_setup.sh`

**Understand current status?**
→ Read `STATUS.md`

**See step-by-step instructions?**
→ Read `docs/PHASE2_WEEK3_EXECUTION_GUIDE.md`

**Check what's done?**
→ Read `docs/PHASE2_WEEK3_COMPLETE.md`

**Understand SIMD implementation?**
→ Read `docs/SIMD_IMPLEMENTATION_COMPLETE.md`

**Apply engine integration?**
→ Read `docs/patches/ENGINE_SIMD_INTEGRATION.patch`

**Verify readiness?**
→ Run `./scripts/verify_simd_integration.sh`

**Check Phase 1 work?**
→ Read `docs/PHASE1_COMPLETION_REPORT.md`

**Review architecture?**
→ Read files in `docs/architecture/`

**Run tests?**
→ Follow `CHECKLIST.md`

---

## 📦 Dependencies

### Managed by Cargo:
See `Cargo.toml` in each crate for specific dependencies.

### Key External Libraries:
- `serde` - Serialization
- `rand` - Random number generation
- `criterion` - Benchmarking
- `thiserror` - Error handling
- `ed25519-dalek` - Cryptography
- `wgpu` - GPU foundation

### SIMD:
- `std::simd` - Portable SIMD (Rust standard library)
- `approx` - Numerical testing

---

## 🎓 Code Organization Principles

### Modularity:
- Each crate < 2000 lines
- Each file < 500 lines
- Clear separation of concerns

### Testing:
- Unit tests inline with code
- Integration tests in `tests/`
- Benchmarks in `benches/`

### Documentation:
- Comprehensive README files
- Inline code documentation
- Separate technical guides

### Automation:
- All repetitive tasks scripted
- Validation at every step
- Performance tracking automated

---

**This structure supports**:
- Easy navigation
- Clear organization
- Automated workflows
- Comprehensive testing
- Complete documentation

**Total project size**: ~90,000 lines of code + documentation
