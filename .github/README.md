# GitHub CI/CD Configuration

This directory contains the complete CI/CD infrastructure for HyperPhysics.

## Workflows

### 1. CI Workflow (`workflows/ci.yml`)

**Triggers:** Push and Pull Requests to `main` and `develop` branches

**Jobs:**

- **Test Suite**
  - Runs on: Linux, macOS, Windows
  - Rust versions: stable, nightly
  - Executes full test suite with caching
  - Build matrix excludes some combinations to optimize CI time

- **Clippy (Lint)**
  - Runs on: Linux
  - Enforces code quality standards
  - Currently set to `continue-on-error: true` for gradual adoption

- **Rustfmt (Format Check)**
  - Runs on: Linux
  - Ensures consistent code formatting
  - Fails build if code is not formatted

- **SIMD Benchmarks**
  - Runs on: Linux (x86_64 only)
  - Validates SIMD performance optimizations
  - Reports CPU capabilities and benchmark results
  - Uses `target-cpu=native` for optimal performance

- **Code Coverage**
  - Runs on: Linux
  - Generates coverage reports using cargo-tarpaulin
  - Uploads to Codecov (if configured)

- **Security Audit**
  - Runs on: Linux
  - Checks for known security vulnerabilities
  - Uses `cargo audit` for dependency scanning

### 2. Release Workflow (`workflows/release.yml`)

**Triggers:** Git tags matching `v*.*.*` (e.g., v0.1.0)

**Jobs:**

- **Create Release**
  - Creates GitHub release from tag

- **Build Release**
  - Builds release binaries for multiple platforms:
    - Linux x86_64
    - macOS x86_64
    - macOS ARM64 (Apple Silicon)
    - Windows x86_64
  - Uploads binaries as release assets

- **Publish to crates.io**
  - Publishes crates in correct dependency order
  - Requires `CARGO_TOKEN` secret to be configured

### 3. Documentation Workflow (`workflows/docs.yml`)

**Triggers:** Push and Pull Requests to `main`

**Jobs:**

- **Build Documentation**
  - Generates rustdoc for all crates
  - Deploys to GitHub Pages on main branch
  - Accessible at: `https://<username>.github.io/<repo>/`

## Configuration Files

### `dependabot.yml`

Automated dependency updates:
- **Cargo dependencies:** Weekly updates
- **GitHub Actions:** Weekly updates
- Auto-assigns reviewers and labels
- Opens up to 10 PRs for Rust deps, 5 for Actions

### `pull_request_template.md`

Standard PR template requiring:
- Description and type of change
- Testing performed checklist
- Code quality verification
- Security considerations

### `CODEOWNERS`

Defines code ownership for different directories:
- Global owner: @fatih-erarslan
- Specific ownership for crypto, market, core modules
- Ensures appropriate reviewers on PRs

## Setup Requirements

### Secrets

Configure these secrets in GitHub repository settings:

1. **CARGO_TOKEN** (required for releases)
   - Generate at https://crates.io/me
   - Used for publishing to crates.io

2. **CODECOV_TOKEN** (optional)
   - From https://codecov.io
   - Used for code coverage reports

### GitHub Pages

Enable GitHub Pages in repository settings:
- Source: gh-pages branch
- Documentation will be auto-deployed on main branch updates

## Caching Strategy

All workflows use aggressive caching to optimize build times:

- **Cargo registry cache:** Shared dependency downloads
- **Cargo index cache:** Package metadata
- **Target directory cache:** Compiled artifacts

Cache keys include:
- OS and Rust version
- Cargo.lock hash for invalidation on dependency changes

## Performance Optimizations

1. **Fail-Fast Strategy:** Disabled to see all platform results
2. **Selective Matrix:** Excludes nightly builds on macOS/Windows
3. **Parallel Jobs:** All independent jobs run in parallel
4. **Continue-on-Error:** Non-critical jobs don't block builds

## Current Status

### ✅ Implemented
- Multi-platform testing
- Multi-toolchain validation
- Clippy linting
- Format checking
- SIMD benchmarking
- Security audits
- Release automation
- Documentation generation
- Dependency updates

### ⚠️ Gradual Adoption Mode

Some checks are set to `continue-on-error: true`:
- Clippy warnings (will be enforced after cleanup)
- Security audit (advisory review period)
- Coverage (informational only)
- Nightly builds (experimental)

### 🔄 Next Steps

1. **Fix Clippy Warnings**
   - Address all clippy warnings in codebase
   - Change `continue-on-error: false` in ci.yml

2. **Enforce Strict Linting**
   - Update workspace Cargo.toml:
     ```toml
     [workspace.lints.rust]
     warnings = "deny"
     ```

3. **Coverage Thresholds**
   - Set minimum coverage requirements
   - Block PRs below threshold

4. **Custom Runners**
   - Consider self-hosted runners for heavy workloads
   - GPU testing on appropriate hardware

## Monitoring

### Build Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/hyperphysics/hyperphysics/workflows/CI/badge.svg)](https://github.com/hyperphysics/hyperphysics/actions/workflows/ci.yml)
[![Release](https://github.com/hyperphysics/hyperphysics/workflows/Release/badge.svg)](https://github.com/hyperphysics/hyperphysics/actions/workflows/release.yml)
[![Docs](https://github.com/hyperphysics/hyperphysics/workflows/Documentation/badge.svg)](https://github.com/hyperphysics/hyperphysics/actions/workflows/docs.yml)
```

### Useful Commands

```bash
# Local validation before pushing
cargo fmt --all -- --check
cargo clippy --workspace --all-features -- -D warnings
cargo test --workspace --all-features

# SIMD benchmarks (x86_64)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench simd_exp

# Security audit
cargo install cargo-audit
cargo audit

# Coverage (Linux)
cargo install cargo-tarpaulin
cargo tarpaulin --workspace --all-features
```

## Troubleshooting

### Common Issues

1. **Build fails on Windows**
   - Check for Unix-specific dependencies
   - Ensure path separators are cross-platform

2. **Nightly builds fail**
   - Expected occasionally
   - Don't block on nightly failures

3. **Cache miss/slow builds**
   - Clear cache in Actions settings
   - Check Cargo.lock is committed

4. **Coverage upload fails**
   - Verify CODECOV_TOKEN is set
   - Check Codecov project configuration

## Maintenance

- **Weekly:** Review Dependabot PRs
- **Monthly:** Check CI usage and costs
- **Quarterly:** Update GitHub Actions versions
- **Annually:** Review and optimize caching strategy

---

**Last Updated:** 2025-11-14
**Maintained By:** @fatih-erarslan
