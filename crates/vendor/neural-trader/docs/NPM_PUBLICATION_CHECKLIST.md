# NPM Publication Checklist - Neural Trader

## Pre-Publication Validation

### 1. Version Management
- [ ] Update version in `/package.json` (currently 2.3.0 → 2.3.1)
- [ ] Update version in `/neural-trader-rust/crates/napi-bindings/Cargo.toml`
- [ ] Update version in `/neural-trader-rust/crates/napi-bindings/package.json`
- [ ] Create CHANGELOG entry for v2.3.1

### 2. Build Validation

```bash
# Build all platform-specific binaries
npm run build:all

# Verify artifacts
npm run artifacts

# Check binaries
npm run check-binaries
```

- [ ] Linux x64 binary built
- [ ] Linux ARM64 binary built
- [ ] macOS x64 binary built
- [ ] macOS ARM64 binary built
- [ ] Windows x64 binary built

### 3. Docker Testing

```bash
cd tests/docker
docker-compose -f docker-compose.test.yml up
```

- [ ] `test-minimal` passes (no build tools)
- [ ] `test-build` passes (with build tools)
- [ ] `test-alpine` passes (musl libc)
- [ ] `test-dependencies` passes (all deps load)

### 4. Installation Testing

```bash
# Create test package
npm pack

# Test installation in clean environment
mkdir /tmp/test-install
cd /tmp/test-install
npm install /path/to/neural-trader-2.3.1.tgz

# Verify installation
npx neural-trader check-binaries
node -e "require('neural-trader')"
```

- [ ] Package installs without errors
- [ ] Binary check passes
- [ ] Module loads successfully
- [ ] CLI works: `npx neural-trader --help`

### 5. Dependency Packages

Check and fix these packages that had installation issues:

#### agentdb (hnswlib-node)
```bash
cd node_modules/agentdb
npm run rebuild  # Should succeed
```
- [ ] agentdb builds successfully
- [ ] hnswlib-node native bindings work

#### aidefence
```bash
cd node_modules/aidefence
ls dist/  # Should contain gateway/server.js
```
- [ ] aidefence has built dist/ folder
- [ ] If not, contact maintainers or fork

#### agentic-payments
```bash
cd node_modules/agentic-payments
ls dist/  # Should contain index.cjs
```
- [ ] agentic-payments has dist/index.cjs
- [ ] If not, contact maintainers or fork

#### sublinear-time-solver
```bash
cd node_modules/sublinear-time-solver
cat package.json | grep exports
```
- [ ] Has proper "exports" field
- [ ] If not, contact maintainers or fork

### 6. Documentation

- [ ] README.md updated with installation instructions
- [ ] INSTALLATION_FIXES.md created
- [ ] NPM_PUBLICATION_CHECKLIST.md created
- [ ] Example code tested and working

### 7. Package Contents Verification

```bash
npm pack --dry-run
```

Check that package includes:
- [ ] `bin/cli.js`
- [ ] `index.js`
- [ ] `scripts/*.js` (install, postinstall, prebuild, check-binaries)
- [ ] `neural-trader-rust/crates/napi-bindings/*.node` (binaries)
- [ ] `neural-trader-rust/crates/napi-bindings/index.js`
- [ ] `neural-trader-rust/crates/napi-bindings/index.d.ts`
- [ ] `packages/core/`
- [ ] `packages/predictor/`
- [ ] `README.md`
- [ ] `LICENSE`

Verify exclusions (.npmignore):
- [ ] `tests/` excluded
- [ ] `docs/` excluded (except README)
- [ ] `.github/` excluded
- [ ] `node_modules/` excluded
- [ ] Source `.rs` files excluded
- [ ] `Cargo.toml` excluded

### 8. Pre-Publish Scripts

Ensure `prepublishOnly` works:

```bash
npm run prepublishOnly
```

- [ ] Builds release binaries
- [ ] Generates artifacts
- [ ] No errors

## Publication Steps

### Step 1: Login to NPM

```bash
npm login
# Enter credentials
npm whoami  # Verify login
```

- [ ] Logged in as correct user
- [ ] Have publish permissions for `neural-trader` package

### Step 2: Dry Run

```bash
npm publish --dry-run
```

- [ ] Review files to be published
- [ ] Verify version number
- [ ] Check package size (should be reasonable)

### Step 3: Publish

```bash
# Publish to NPM
npm publish

# Or for scoped packages with public access:
# npm publish --access public
```

- [ ] Published successfully
- [ ] No errors during upload

### Step 4: Verify Publication

```bash
# Wait 1-2 minutes for NPM to propagate
npm view neural-trader

# Test installation from NPM
mkdir /tmp/test-npm
cd /tmp/test-npm
npm install neural-trader@2.3.1
npx neural-trader check-binaries
```

- [ ] Package appears on npm
- [ ] Correct version shows
- [ ] Installation works from npm
- [ ] Binaries are included

### Step 5: Git Tagging

```bash
git add .
git commit -m "fix: resolve installation errors and missing binaries (v2.3.1)

- Add missing install.js script
- Fix NAPI bindings packaging
- Create Python fallback setup
- Add comprehensive Docker tests
- Fix package.json files field
- Add postinstall and prebuild scripts
- Add binary validation tools

Fixes: #XXX"

git tag -a v2.3.1 -m "Release v2.3.1 - Installation Fixes"
git push origin fix/installation-binaries-missing
git push origin v2.3.1
```

- [ ] Committed all changes
- [ ] Tagged release
- [ ] Pushed to remote

### Step 6: GitHub Release

Create GitHub release:
- [ ] Go to https://github.com/ruvnet/neural-trader/releases/new
- [ ] Tag: `v2.3.1`
- [ ] Title: `v2.3.1 - Installation Fixes`
- [ ] Description: Copy from CHANGELOG
- [ ] Attach binary artifacts (optional)
- [ ] Publish release

### Step 7: Update Documentation

- [ ] Update main README if needed
- [ ] Update installation documentation
- [ ] Close related issues
- [ ] Announce in discussions/Discord/Slack

## Post-Publication

### Monitor for Issues

- [ ] Check npm downloads: `npm info neural-trader`
- [ ] Monitor GitHub issues for installation problems
- [ ] Check CI/CD pipelines
- [ ] Test on different platforms

### Platform-Specific Package Publishing (Optional)

If using platform-specific optional dependencies:

```bash
# Publish each platform package separately
cd neural-trader-rust/crates/napi-bindings

# Linux x64
npm publish --tag linux-x64

# macOS ARM64
npm publish --tag darwin-arm64

# etc.
```

## Rollback Plan

If issues are discovered:

```bash
# Deprecate the bad version
npm deprecate neural-trader@2.3.1 "Installation issues, use v2.3.2"

# Publish fixed version
# Update to 2.3.2
npm publish
```

## Related Package Updates

If these packages need updates:

### @neural-trader/core
```bash
cd packages/core
npm version patch
npm publish
```

### @neural-trader/predictor
```bash
cd packages/predictor
npm version patch
npm publish
```

### Example packages
```bash
cd packages/examples
npm run publish:all
```

## Success Criteria

✅ Package published without errors
✅ All Docker tests pass
✅ Installation works on all supported platforms
✅ Binaries load correctly
✅ No regression in existing functionality
✅ Dependencies resolve correctly
✅ CLI works as expected

## Support Preparation

After publication:
- [ ] Update FAQ with common installation issues
- [ ] Prepare support responses for known issues
- [ ] Monitor npm support channels
- [ ] Be ready for quick patch if needed

## Notes

- Always test thoroughly before publishing
- Consider publishing to a test registry first
- Keep communication channels open for user feedback
- Document any platform-specific limitations
- Maintain backwards compatibility

## Emergency Contacts

- NPM Support: https://www.npmjs.com/support
- Package maintainers: [list emails]
- CI/CD admin: [contact info]
