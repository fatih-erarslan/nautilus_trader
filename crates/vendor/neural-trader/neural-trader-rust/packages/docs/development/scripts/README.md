# Development Scripts

Automation scripts for building, testing, publishing, and validating packages.

## 📚 Available Scripts

### Publishing
- **[publish-all-packages.sh](./publish-all-packages.sh)** - Publish all packages to NPM
  - Validates packages first
  - Builds all packages
  - Publishes to NPM registry
  - Verifies publication
  - Handles errors gracefully

### Validation
- **[validate-all-packages.sh](./validate-all-packages.sh)** - Validate all packages
  - Checks package.json validity
  - Verifies dependencies
  - Tests imports
  - Runs test suites
  - Checks build output

## 🚀 Quick Start

### Publish All Packages
```bash
# From repository root
./packages/docs/development/scripts/publish-all-packages.sh

# Or from scripts directory
cd packages/docs/development/scripts
./publish-all-packages.sh
```

**What it does:**
1. Validates all packages
2. Builds native bindings for all platforms
3. Runs tests
4. Publishes to NPM (requires login)
5. Verifies packages are accessible
6. Generates summary report

### Validate All Packages
```bash
# From repository root
./packages/docs/development/scripts/validate-all-packages.sh

# Or from scripts directory
cd packages/docs/development/scripts
./validate-all-packages.sh
```

**What it checks:**
1. package.json structure
2. Dependencies installed
3. Import validation
4. Type definitions
5. Build artifacts
6. Test suite status

## 📦 Script Details

### publish-all-packages.sh

**Prerequisites:**
- NPM account with @neural-trader scope access
- Logged in via `npm login`
- All packages built and tested

**Usage:**
```bash
./publish-all-packages.sh [options]

Options:
  --dry-run     Show what would be published without publishing
  --skip-tests  Skip test suite execution
  --force       Force publish even if validation fails
```

**Output:**
- Success/failure for each package
- NPM registry URLs
- Version numbers
- Total packages published
- Summary report

**Error Handling:**
- Stops on critical errors
- Continues on warnings
- Logs all errors to file
- Provides rollback instructions

### validate-all-packages.sh

**Prerequisites:**
- Node.js 16+ installed
- Dependencies installed (`npm install`)

**Usage:**
```bash
./validate-all-packages.sh [options]

Options:
  --verbose     Show detailed validation output
  --fix         Attempt to fix issues automatically
  --report      Generate detailed validation report
```

**Validation Checks:**
1. **Structure**
   - package.json exists
   - Required fields present
   - Version format valid

2. **Dependencies**
   - All deps installed
   - Peer deps satisfied
   - No circular deps

3. **Functionality**
   - Package imports
   - Core exports work
   - No runtime errors

4. **Build**
   - Native bindings present (if needed)
   - Type definitions included
   - Build artifacts valid

**Output:**
- ✅ Pass
- ⚠️ Warning
- ❌ Fail
- Summary statistics

## 🔧 Script Customization

### Environment Variables

```bash
# Publishing configuration
NPM_REGISTRY="https://registry.npmjs.org"
NPM_SCOPE="@neural-trader"

# Validation settings
SKIP_BUILD_CHECK=false
SKIP_TESTS=false
VERBOSE=false

# Paths
PACKAGES_DIR="../../../packages"
LOGS_DIR="./logs"
```

### Adding New Scripts

1. Create script in `scripts/` directory
2. Make executable: `chmod +x script-name.sh`
3. Add documentation to this README
4. Update related documentation

## 📊 Script Workflow

### Publishing Workflow
```
1. Validate → 2. Build → 3. Test → 4. Publish → 5. Verify → 6. Report
     ↓            ↓          ↓          ↓           ↓           ↓
  validate.sh   npm build  npm test  npm publish  npm view   summary
```

### Validation Workflow
```
1. Check Structure → 2. Check Dependencies → 3. Test Import → 4. Run Tests → 5. Report
        ↓                    ↓                      ↓              ↓           ↓
   package.json         npm list            require(pkg)      npm test    results
```

## 🐛 Troubleshooting

### Common Issues

**Issue: Permission Denied**
```bash
# Make scripts executable
chmod +x *.sh
```

**Issue: NPM Authentication Failed**
```bash
# Login to NPM
npm login
```

**Issue: Package Validation Failed**
```bash
# Run with verbose output
./validate-all-packages.sh --verbose

# Check specific package
cd packages/<package-name>
npm test
```

**Issue: Build Failed**
```bash
# Clean and rebuild
npm run clean
npm run build:all
```

## 📈 Best Practices

### Before Publishing
1. Run validation script
2. Fix all critical issues
3. Review changelogs
4. Bump versions appropriately
5. Test locally
6. Commit changes
7. Run publish script

### Automation
- Use in CI/CD pipelines
- Schedule regular validation
- Monitor NPM registry
- Track download statistics

### Error Handling
- Always check exit codes
- Log errors to files
- Provide clear error messages
- Include recovery steps

## 🔗 Related Documentation

- [Publishing Documentation](../publishing/) - Publishing workflow
- [Testing Documentation](../testing/) - Test suite
- [Build Documentation](../build/) - Build system

---

[← Back to Development](../README.md)
