# Publishing Checklist

This checklist ensures a smooth and error-free publishing process for Neural Trader packages.

## Pre-Publishing Phase

### 1. Code Quality & Testing
- [ ] All unit tests pass: `cargo test --workspace`
- [ ] All integration tests pass: `cargo test --all-features`
- [ ] Clippy passes with no warnings: `cargo clippy --all-targets --all-features -- -D warnings`
- [ ] Code is properly formatted: `cargo fmt --all -- --check`
- [ ] NPM tests pass: `npm test`
- [ ] Manual testing completed for critical features
- [ ] No TODO or FIXME comments in release code

### 2. Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md has entry for new version
- [ ] API documentation is complete: `cargo doc --no-deps`
- [ ] Examples are working and up to date
- [ ] Migration guide written (if breaking changes)
- [ ] Release notes drafted

### 3. Dependencies
- [ ] All dependencies are up to date
- [ ] No security vulnerabilities: `cargo audit`
- [ ] NPM dependencies audited: `npm audit`
- [ ] Dependency versions are consistent across workspace
- [ ] No unused dependencies

### 4. Version Management
- [ ] Version number follows semver
- [ ] Version is consistent across all packages
- [ ] Git is clean (no uncommitted changes)
- [ ] All changes are committed
- [ ] Working on correct branch (main/release)

### 5. Configuration
- [ ] Cargo.toml metadata is correct (authors, license, repository)
- [ ] package.json metadata is correct
- [ ] .npmignore and .gitignore are properly configured
- [ ] License files are present and correct
- [ ] README and documentation are included in packages

## Publishing Phase

### 6. Bump Version
```bash
./scripts/bump-version.sh <new-version> --dry-run  # Test first
./scripts/bump-version.sh <new-version>            # Execute
```
- [ ] Version bumped successfully
- [ ] CHANGELOG.md updated
- [ ] Git tag created
- [ ] Changes committed

### 7. Push to GitHub
```bash
git push origin main
git push origin v<version>
```
- [ ] Code pushed to main branch
- [ ] Tag pushed to GitHub
- [ ] CI/CD pipeline triggered
- [ ] All CI checks passing

### 8. Publish Rust Crates
```bash
./scripts/publish-crates.sh --dry-run  # Test first
./scripts/publish-crates.sh            # Execute
```
- [ ] All crates published in correct order
- [ ] No publish failures
- [ ] Crates.io pages look correct
- [ ] Documentation on docs.rs is building

### 9. Build NAPI Binaries
```bash
npm run build  # Or wait for GitHub Actions
```
- [ ] Binaries built for all platforms:
  - [ ] x86_64-unknown-linux-gnu
  - [ ] aarch64-unknown-linux-gnu
  - [ ] x86_64-apple-darwin
  - [ ] aarch64-apple-darwin
  - [ ] x86_64-pc-windows-msvc
- [ ] Binaries tested on each platform
- [ ] No runtime errors

### 10. Publish NPM Packages
```bash
./scripts/publish-npm.sh --dry-run  # Test first
./scripts/publish-npm.sh            # Execute
```
- [ ] All packages published in correct order
- [ ] No publish failures
- [ ] NPM package pages look correct
- [ ] Installation works: `npm install neural-trader@<version>`

### 11. Create GitHub Release
- [ ] GitHub release created (automated or manual)
- [ ] Release notes are complete
- [ ] Binaries attached to release
- [ ] Release is not marked as draft
- [ ] Release is not marked as pre-release (unless it is)

## Post-Publishing Phase

### 12. Verification
- [ ] Install from crates.io: `cargo install neural-trader-cli`
- [ ] Install from npm: `npm install -g neural-trader`
- [ ] Test basic functionality
- [ ] Check documentation links work
- [ ] Verify examples work with new version

### 13. Announcements
- [ ] Update project website (if applicable)
- [ ] Post on social media (Twitter, LinkedIn)
- [ ] Update Discord/Slack announcements
- [ ] Send newsletter (if applicable)
- [ ] Update project status page

### 14. Monitoring
- [ ] Monitor download statistics
- [ ] Watch for issue reports
- [ ] Check error tracking (Sentry, etc.)
- [ ] Review user feedback
- [ ] Monitor CI/CD pipeline

## Rollback Procedure (If Needed)

If issues are discovered after publishing:

### 1. Assess Severity
- Critical bug or security issue? → Immediate rollback
- Minor issue? → Patch release

### 2. Execute Rollback
```bash
./scripts/rollback.sh <version> --confirm
```
- [ ] Crates yanked from crates.io
- [ ] NPM packages deprecated
- [ ] GitHub release deleted
- [ ] Git tag removed
- [ ] Commits reverted

### 3. Fix and Re-publish
- [ ] Fix the issue
- [ ] Bump to new patch version
- [ ] Run through checklist again
- [ ] Publish fixed version

## Troubleshooting

### Common Issues

#### Crate Already Published
- Check if version was already published
- Bump to next version if needed
- Don't yank unless critical issue

#### NPM Publish Fails
- Check npm authentication: `npm whoami`
- Verify package name is not taken
- Check for missing files in package

#### GitHub Actions Fails
- Check workflow logs
- Verify secrets are set (CARGO_REGISTRY_TOKEN, NPM_TOKEN)
- Test locally with act: `act -j build-napi`

#### Binary Compatibility Issues
- Verify NAPI version compatibility
- Check Node.js version support
- Test on all platforms

## Scripts Reference

### bump-version.sh
```bash
./scripts/bump-version.sh <version> [--dry-run]
```
Updates version across all packages and creates git tag.

### publish-crates.sh
```bash
./scripts/publish-crates.sh [--dry-run] [--skip-tests]
```
Publishes Rust crates in dependency order.

### publish-npm.sh
```bash
./scripts/publish-npm.sh [--dry-run] [--skip-build]
```
Publishes NPM packages with NAPI binaries.

### rollback.sh
```bash
./scripts/rollback.sh <version> [--confirm]
```
Rolls back a published version.

## Security Checklist

- [ ] No secrets in code or config files
- [ ] API keys are stored securely
- [ ] Dependencies have no known vulnerabilities
- [ ] Security audit completed
- [ ] Sensitive data is properly sanitized

## Performance Checklist

- [ ] Benchmarks show no regressions
- [ ] Memory usage is acceptable
- [ ] Binary size is reasonable
- [ ] Startup time is fast
- [ ] No performance warnings

## Compliance Checklist

- [ ] License is correct and compatible
- [ ] Third-party licenses are acknowledged
- [ ] Copyright notices are present
- [ ] Terms of service are followed
- [ ] GDPR compliance (if applicable)

## Notes

- Always test with `--dry-run` first
- Keep CHANGELOG.md up to date
- Communicate breaking changes clearly
- Monitor for issues after publishing
- Be prepared to rollback if needed

## Useful Commands

```bash
# Check current version
grep -m1 '^version' Cargo.toml

# View git tags
git tag -l

# Check crates.io status
cargo search neural-trader

# Check npm status
npm view neural-trader

# View GitHub releases
gh release list

# Test installation
cargo install --force neural-trader-cli
npm install -g neural-trader@latest
```

## Support

If you encounter issues during publishing:
1. Check this checklist
2. Review script logs in publish-*.log files
3. Check GitHub Actions logs
4. Consult team members
5. Use rollback script if needed

---

**Remember:** Better to delay a release than to ship a broken one!
