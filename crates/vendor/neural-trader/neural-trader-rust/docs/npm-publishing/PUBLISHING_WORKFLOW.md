# Publishing Workflow Summary

Complete guide to publishing Neural Trader Rust crates and NPM packages.

## Overview

The publishing workflow consists of:
1. **Version Management** - Bump versions across all packages
2. **Rust Crates Publishing** - Publish to crates.io in dependency order
3. **NPM Packages Publishing** - Build NAPI binaries and publish to npm
4. **GitHub Release** - Create release with binaries
5. **Documentation Updates** - Deploy docs to GitHub Pages

## Quick Start

### Prerequisites

```bash
# Install required tools
cargo install cargo-audit
npm install -g npm

# Authenticate
cargo login <your-crates-io-token>
npm login

# Install GitHub CLI (optional, for releases)
# See: https://cli.github.com/
```

### Publishing Process

```bash
# 1. Bump version
./scripts/bump-version.sh 1.0.8

# 2. Review changes
git diff

# 3. Commit and push
git add -A
git commit -m "chore: bump version to 1.0.8"
git push origin main
git push origin v1.0.8

# 4. Publish Rust crates
./scripts/publish-crates.sh

# 5. Publish NPM packages (after GitHub Actions builds binaries)
./scripts/publish-npm.sh

# Or let GitHub Actions handle everything:
# Push tag and wait for automated workflow
```

## Scripts Location

All publishing scripts are located in `/workspaces/neural-trader/neural-trader-rust/scripts/`:

- **bump-version.sh** - Version management
- **publish-crates.sh** - Rust crates publishing
- **publish-npm.sh** - NPM packages publishing
- **rollback.sh** - Emergency rollback

## Detailed Workflows

### 1. Version Management

**Script:** `scripts/bump-version.sh`

**Purpose:** Update version numbers consistently across all packages.

**Usage:**
```bash
# Dry run (recommended first)
./scripts/bump-version.sh 1.0.8 --dry-run

# Execute
./scripts/bump-version.sh 1.0.8
```

**What it does:**
- Updates all `Cargo.toml` files
- Updates all `package.json` files
- Updates internal dependencies
- Generates CHANGELOG.md entry
- Creates git tag
- Updates package-lock.json

**Version Format:** Must follow semver: `MAJOR.MINOR.PATCH[-PRERELEASE]`

**Examples:**
- `1.0.8` - Patch release
- `1.1.0` - Minor release with new features
- `2.0.0` - Major release with breaking changes
- `1.0.8-beta.1` - Pre-release

### 2. Rust Crates Publishing

**Script:** `scripts/publish-crates.sh`

**Purpose:** Publish all Rust crates to crates.io in correct dependency order.

**Usage:**
```bash
# Dry run
./scripts/publish-crates.sh --dry-run

# Execute
./scripts/publish-crates.sh

# Skip tests (not recommended)
./scripts/publish-crates.sh --skip-tests
```

**Publishing Order:**
1. **Base crate:** `nt-core` (no dependencies)
2. **Feature crates** (parallel):
   - `nt-strategies`
   - `nt-neural`
   - `nt-risk`
   - `nt-portfolio`
   - `nt-backtest`
   - `nt-execution`
   - `nt-data`
   - `nt-sentiment`
   - `nt-prediction`
   - `nt-sports`
   - `nt-syndicate`
   - `nt-e2b`
3. **NAPI binding:** `nt-napi` (depends on all above)

**Features:**
- Checks if version already published
- Runs tests before publishing
- Validates package contents
- Waits for crates.io index updates
- Comprehensive error handling
- Detailed logging

**Wait Times:**
- Base crates: 45 seconds
- NAPI crate: 60 seconds (larger, needs more time)

### 3. NPM Packages Publishing

**Script:** `scripts/publish-npm.sh`

**Purpose:** Publish all NPM packages with pre-built NAPI binaries.

**Usage:**
```bash
# Dry run
./scripts/publish-npm.sh --dry-run

# Execute
./scripts/publish-npm.sh

# Skip building (use pre-built binaries)
./scripts/publish-npm.sh --skip-build
```

**Publishing Order:**
1. `@neural-trader/mcp-protocol` - Protocol definitions
2. `@neural-trader/mcp` - MCP server implementation
3. `neural-trader` - Meta package with NAPI bindings

**NAPI Binary Platforms:**
- `x86_64-unknown-linux-gnu` - Linux x64
- `aarch64-unknown-linux-gnu` - Linux ARM64
- `x86_64-apple-darwin` - macOS Intel
- `aarch64-apple-darwin` - macOS Apple Silicon
- `x86_64-pc-windows-msvc` - Windows x64

**Features:**
- Checks npm authentication
- Verifies package versions
- Runs prepublishOnly hooks
- Validates package contents
- Creates GitHub release
- Comprehensive error handling

### 4. GitHub Actions Workflow

**File:** `.github/workflows/publish.yml`

**Triggers:**
- Push tags matching `v*.*.*`
- Manual workflow dispatch

**Jobs:**

1. **test** - Run full test suite
   - Run cargo tests
   - Run clippy
   - Check formatting

2. **build-napi** - Build binaries for all platforms
   - Matrix strategy for 5 platforms
   - Upload artifacts for each platform
   - Parallel execution

3. **publish-crates** - Publish Rust crates
   - Uses `publish-crates.sh`
   - Requires `CARGO_REGISTRY_TOKEN` secret

4. **publish-npm** - Publish NPM packages
   - Downloads NAPI artifacts
   - Uses `publish-npm.sh`
   - Requires `NPM_TOKEN` secret

5. **create-release** - Create GitHub release
   - Extract changelog
   - Attach NAPI binaries
   - Use `GITHUB_TOKEN`

6. **update-docs** - Deploy documentation
   - Generate cargo docs
   - Deploy to GitHub Pages

**Required Secrets:**
- `CARGO_REGISTRY_TOKEN` - crates.io API token
- `NPM_TOKEN` - npm automation token
- `GITHUB_TOKEN` - automatic, provided by GitHub

### 5. Rollback Procedures

**Script:** `scripts/rollback.sh`

**Purpose:** Emergency rollback of published versions.

**Usage:**
```bash
# Dry run
./scripts/rollback.sh 1.0.8

# Execute (requires confirmation)
./scripts/rollback.sh 1.0.8 --confirm
```

**What it does:**
- Yanks all Rust crates from crates.io
- Deprecates all NPM packages
- Deletes GitHub release
- Removes git tags (local and remote)
- Reverts git commits

**When to use:**
- Critical bugs discovered after publishing
- Security vulnerabilities found
- Breaking changes in patch release
- Accidental publish of wrong version

**Note:** Yanking/deprecating doesn't delete packages, just marks them as problematic.

## Testing Recommendations

### Pre-Publishing Tests

```bash
# 1. Full test suite
cargo test --all-features --workspace

# 2. Clippy (strict mode)
cargo clippy --all-targets --all-features -- -D warnings

# 3. Format check
cargo fmt --all -- --check

# 4. Security audit
cargo audit

# 5. Build for release
cargo build --release

# 6. NPM tests
npm test

# 7. NPM audit
npm audit

# 8. Manual testing
cargo run --example basic_strategy
npx neural-trader ping
```

### Dry Run Tests

```bash
# 1. Test version bump
./scripts/bump-version.sh 1.0.8 --dry-run

# 2. Test crates publishing
./scripts/publish-crates.sh --dry-run

# 3. Test NPM publishing
./scripts/publish-npm.sh --dry-run

# 4. Test rollback
./scripts/rollback.sh 1.0.8
```

### Post-Publishing Verification

```bash
# 1. Install from crates.io
cargo install --force neural-trader-cli

# 2. Install from npm
npm install -g neural-trader@1.0.8

# 3. Test basic functionality
neural-trader --version
neural-trader ping

# 4. Check documentation
open https://docs.rs/nt-core/1.0.8

# 5. Verify GitHub release
gh release view v1.0.8
```

## Common Issues & Solutions

### Issue: Crate Already Published
**Solution:** Version is already on crates.io. Either:
- Skip if intentional
- Bump to next version
- Yank and re-publish with new version

### Issue: NPM Authentication Failed
**Solution:**
```bash
npm whoami  # Check if logged in
npm login   # Log in if needed
```

### Issue: GitHub Actions Fails
**Solutions:**
- Check workflow logs on GitHub
- Verify secrets are set correctly
- Test locally with `act`
- Review recent changes for syntax errors

### Issue: Binary Compatibility Error
**Solutions:**
- Rebuild with correct Node.js version
- Check NAPI version compatibility
- Verify platform target is correct
- Test on actual target platform

### Issue: Crates.io Index Not Updated
**Solution:** Wait longer (up to 5 minutes) and retry

### Issue: Tests Fail During Publishing
**Solution:**
- Fix failing tests first
- Don't use `--skip-tests` unless absolutely necessary
- Re-run after fixes

## Best Practices

### Before Publishing
1. Always run dry-run first
2. Test on multiple platforms
3. Review CHANGELOG.md
4. Update documentation
5. Check for breaking changes
6. Ensure clean git state

### During Publishing
1. Publish during low-traffic hours
2. Monitor logs in real-time
3. Keep terminal open until complete
4. Don't interrupt the process
5. Have rollback plan ready

### After Publishing
1. Verify installations work
2. Test on fresh systems
3. Monitor error reports
4. Update announcements
5. Watch download statistics
6. Be ready to patch if needed

## Emergency Contacts

If publishing fails critically:
1. Stop the process immediately
2. Run rollback script
3. Review logs in publish-*.log files
4. Contact team lead
5. Document the issue
6. Fix before re-attempting

## Monitoring & Metrics

### What to Monitor
- Download statistics (crates.io, npm)
- Issue reports on GitHub
- Error tracking (if configured)
- User feedback (Discord, forums)
- Build status (GitHub Actions)

### Key Metrics
- Time to publish
- Success rate
- Download count
- Issue rate
- User satisfaction

## Automation

### Fully Automated Publishing
To enable fully automated publishing on tag push:
1. Set up required secrets in GitHub
2. Push version tag: `git push origin v1.0.8`
3. GitHub Actions handles everything
4. Monitor workflow progress
5. Verify successful completion

### Semi-Automated Publishing
For more control:
1. Run scripts manually
2. Review each step
3. Proceed only if successful
4. Handle issues immediately

## Documentation Links

- **Crates.io:** https://crates.io/crates/neural-trader
- **NPM:** https://npmjs.com/package/neural-trader
- **Docs.rs:** https://docs.rs/nt-core
- **GitHub:** https://github.com/ruvnet/neural-trader
- **Cargo Book:** https://doc.rust-lang.org/cargo/
- **NPM Docs:** https://docs.npmjs.com/

## Support

For help with publishing:
1. Check this documentation
2. Review PUBLISHING_CHECKLIST.md
3. Check script logs
4. Review GitHub Actions logs
5. Open issue on GitHub

---

**Remember:** Publishing is permanent. Test thoroughly before releasing!
