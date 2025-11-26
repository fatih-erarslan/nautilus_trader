# Release Checklist

This document outlines the process for releasing a new version of Neural Trader Rust.

## Pre-Release Preparation

### 1. Code Quality

- [ ] All tests passing locally (`cargo test --workspace`)
- [ ] All benchmarks passing (`cargo bench --workspace`)
- [ ] No clippy warnings (`cargo clippy --all-targets --all-features -- -D warnings`)
- [ ] Code formatted (`cargo fmt --all -- --check`)
- [ ] Documentation builds (`cargo doc --no-deps --workspace`)
- [ ] Examples run successfully

### 2. Security & Compliance

- [ ] Security audit clean (`cargo audit`)
- [ ] License compliance verified (`cargo deny check licenses`)
- [ ] Dependency advisories checked (`cargo deny check advisories`)
- [ ] No known CVEs in dependencies
- [ ] Secrets scanning passed (GitHub Advanced Security)
- [ ] SECURITY.md is up to date

### 3. Performance Validation

- [ ] Benchmarks meet performance targets:
  - [ ] Strategy execution: <200ms
  - [ ] Risk calculation: <50ms
  - [ ] Portfolio rebalancing: <100ms
  - [ ] Market data ingestion: <10ms
  - [ ] Order execution: <150ms
- [ ] Memory usage acceptable (<100MB baseline)
- [ ] No memory leaks detected (`valgrind` or `heaptrack`)
- [ ] CPU usage within limits (<30% idle, <80% under load)

### 4. Documentation

- [ ] CHANGELOG.md updated with all changes
- [ ] README.md accurate and complete
- [ ] API documentation complete (rustdoc)
- [ ] Migration guide written (if breaking changes)
- [ ] Configuration examples updated
- [ ] Tutorial/getting started guide reviewed

### 5. Version Management

- [ ] Version bumped in all `Cargo.toml` files (workspace version)
- [ ] Version bumped in `package.json`
- [ ] Version bumped in all npm platform packages
- [ ] Git tags follow semantic versioning (vX.Y.Z)
- [ ] Version documented in CHANGELOG.md

### 6. Integration Testing

- [ ] Full end-to-end tests pass (`cargo test --test test_full_trading_loop`)
- [ ] Docker build successful (`docker build -t neural-trader .`)
- [ ] Docker smoke test passes
- [ ] Node.js bindings work (`npm test`)
- [ ] CLI commands functional (`neural-trader --help`)

### 7. Cross-Platform Validation

- [ ] Linux x86_64 build successful
- [ ] macOS x86_64 build successful
- [ ] macOS ARM64 build successful
- [ ] Windows x64 build successful
- [ ] Linux musl build successful

### 8. CI/CD Pipeline

- [ ] All GitHub Actions workflows passing
- [ ] Test coverage meets threshold (>80%)
- [ ] Benchmarks show no regression
- [ ] Security scans clean
- [ ] License checks pass

## Release Process

### 1. Create Release Branch

```bash
git checkout -b release/v0.1.0
```

### 2. Update Version Numbers

```bash
# Update all Cargo.toml files
find . -name "Cargo.toml" -exec sed -i 's/version = "0.0.9"/version = "0.1.0"/g' {} \;

# Update package.json
sed -i 's/"version": "0.0.9"/"version": "0.1.0"/g' package.json

# Update npm packages
find npm -name "package.json" -exec sed -i 's/"version": "0.0.9"/"version": "0.1.0"/g' {} \;
```

### 3. Update CHANGELOG.md

```markdown
## [0.1.0] - 2024-XX-XX

### Added
- New momentum strategy implementation
- AgentDB self-learning integration
- Real-time risk management

### Changed
- Improved portfolio rebalancing algorithm
- Enhanced error handling

### Fixed
- Race condition in order execution
- Memory leak in market data streaming

### Performance
- 2x faster strategy execution
- 30% reduction in memory usage
```

### 4. Commit Changes

```bash
git add .
git commit -m "chore: bump version to 0.1.0"
git push origin release/v0.1.0
```

### 5. Create Pull Request

- [ ] Create PR from `release/v0.1.0` to `main`
- [ ] Request reviews from maintainers
- [ ] Wait for all checks to pass
- [ ] Address review feedback
- [ ] Merge PR (use "Create a merge commit")

### 6. Create Git Tag

```bash
git checkout main
git pull origin main
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

### 7. Create GitHub Release

- [ ] Go to https://github.com/ruvnet/neural-trader/releases/new
- [ ] Select tag: `v0.1.0`
- [ ] Release title: `Neural Trader v0.1.0`
- [ ] Copy CHANGELOG.md content to description
- [ ] Mark as pre-release if applicable
- [ ] Publish release

### 8. Automated Publishing (GitHub Actions)

Once the release is created, GitHub Actions will automatically:

- [ ] Build release binaries for all platforms
- [ ] Build Node.js native modules
- [ ] Publish to npm as `@neural-trader/core`
- [ ] Build and push Docker image to Docker Hub
- [ ] Upload release artifacts to GitHub

Monitor the workflow at: https://github.com/ruvnet/neural-trader/actions

### 9. Manual Publishing (if needed)

#### Publish to crates.io

```bash
# Login to crates.io
cargo login <your-token>

# Publish crates in dependency order
cd crates/core && cargo publish
cd ../market-data && cargo publish
cd ../features && cargo publish
cd ../strategies && cargo publish
# ... continue for all crates
```

#### Publish to npm

```bash
# Build all native modules
npm run build

# Collect artifacts
npm run artifacts

# Publish to npm
npm publish --access public
```

#### Push Docker Image

```bash
# Build image
docker build -t neuraltrader/neural-trader-rust:0.1.0 .
docker tag neuraltrader/neural-trader-rust:0.1.0 neuraltrader/neural-trader-rust:latest

# Login to Docker Hub
docker login

# Push images
docker push neuraltrader/neural-trader-rust:0.1.0
docker push neuraltrader/neural-trader-rust:latest
```

## Post-Release Verification

### 1. Installation Testing

```bash
# Test npm package
npm install @neural-trader/core@0.1.0
node -e "const nt = require('@neural-trader/core'); console.log(nt.version);"

# Test cargo installation
cargo install neural-trader-cli --version 0.1.0
neural-trader --version

# Test Docker image
docker pull neuraltrader/neural-trader-rust:0.1.0
docker run neuraltrader/neural-trader-rust:0.1.0 --version
```

### 2. Smoke Tests

- [ ] CLI `neural-trader init` creates project
- [ ] CLI `neural-trader backtest` runs successfully
- [ ] Node.js bindings work with sample code
- [ ] Docker container starts and responds to health checks
- [ ] API endpoints return expected responses

### 3. Production Monitoring

- [ ] Deploy to staging environment
- [ ] Run production validation suite
- [ ] Monitor error rates (should be <0.1%)
- [ ] Monitor latency (should meet SLAs)
- [ ] Check memory usage (should be stable)
- [ ] Verify no regressions in key metrics

### 4. Documentation Updates

- [ ] Update documentation website
- [ ] Publish blog post announcing release
- [ ] Update Discord/Slack announcement
- [ ] Send newsletter to users
- [ ] Update examples repository

### 5. Tracking and Communication

- [ ] Create milestone for next release
- [ ] Move open issues to next milestone
- [ ] Close release tracking issue
- [ ] Thank contributors in release notes
- [ ] Announce on social media (Twitter, LinkedIn, Reddit)

## Hotfix Process

For critical bugs requiring immediate patch:

### 1. Create Hotfix Branch

```bash
git checkout -b hotfix/v0.1.1 v0.1.0
```

### 2. Fix the Bug

- [ ] Identify root cause
- [ ] Write test that reproduces bug
- [ ] Fix bug
- [ ] Verify test passes
- [ ] Add regression test

### 3. Update Version

```bash
# Bump patch version only
# Update Cargo.toml files
# Update package.json
# Update CHANGELOG.md
```

### 4. Release Hotfix

```bash
git commit -m "fix: critical bug in order execution"
git push origin hotfix/v0.1.1

# Create PR to main
# After merge, create tag
git tag -a v0.1.1 -m "Hotfix release 0.1.1"
git push origin v0.1.1
```

### 5. Backport to Stable Branches

If maintaining multiple versions:

```bash
git checkout release/v0.0.x
git cherry-pick <commit-hash>
git tag -a v0.0.10 -m "Backport fix to v0.0.x"
git push origin v0.0.10
```

## Rollback Procedure

If a release has critical issues:

### 1. Immediate Actions

- [ ] Mark release as broken in documentation
- [ ] Post announcement warning users
- [ ] Revert Docker image tags to previous version
- [ ] Deprecate npm version (`npm deprecate @neural-trader/core@0.1.0 "Critical bug, use 0.0.9"`)

### 2. Technical Rollback

```bash
# Revert Git tag
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0

# Create rollback release
git revert <bad-commit>
git tag -a v0.1.1 -m "Rollback of v0.1.0"
git push origin v0.1.1
```

### 3. Communication

- [ ] GitHub release notes explaining rollback
- [ ] Email to affected users
- [ ] Post-mortem document
- [ ] Action items to prevent recurrence

## Release Schedule

- **Major releases** (X.0.0): Every 6 months
- **Minor releases** (0.X.0): Every 4-6 weeks
- **Patch releases** (0.0.X): As needed for critical bugs
- **Pre-releases** (0.X.0-beta.Y): 2 weeks before major/minor releases

## Support Policy

- **Latest major version**: Full support (new features + bug fixes)
- **Previous major version**: Bug fixes + security patches (12 months)
- **Older versions**: Security patches only (if severe)

## Contacts

- **Release Manager**: [Your Name] (your.email@neural-trader.io)
- **Security Contact**: security@neural-trader.io
- **Infrastructure**: devops@neural-trader.io

## Additional Resources

- Semantic Versioning: https://semver.org/
- Keep a Changelog: https://keepachangelog.com/
- Rust Release Process: https://forge.rust-lang.org/release/process.html
- npm Publishing Guide: https://docs.npmjs.com/cli/publish
