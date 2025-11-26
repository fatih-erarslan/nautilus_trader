# NPM Publishing Guide - Neural Trader

## Overview

This guide documents the NPM publishing process for the Neural Trader project, which consists of 16 packages published to the npm registry.

## Package Architecture

### Dependency Hierarchy

```
neural-trader (meta package)
├── @neural-trader/core (foundation)
├── @neural-trader/mcp-protocol (protocol definitions)
├── @neural-trader/mcp (MCP server)
│   ├── depends on: core, mcp-protocol
└── Feature Packages (all depend on core)
    ├── @neural-trader/backtesting
    ├── @neural-trader/brokers
    ├── @neural-trader/execution
    ├── @neural-trader/features
    ├── @neural-trader/market-data
    ├── @neural-trader/neural
    ├── @neural-trader/news-trading
    ├── @neural-trader/portfolio
    ├── @neural-trader/prediction-markets
    ├── @neural-trader/risk
    ├── @neural-trader/sports-betting
    └── @neural-trader/strategies
```

## Publishing Process

### Prerequisites

1. **NPM Authentication**
   ```bash
   npm whoami  # Verify logged in
   npm login   # If not authenticated
   ```

2. **Build Core Package**
   ```bash
   cd packages/core
   npm run build
   ```

3. **Verify Package Names**
   ```bash
   npm view @neural-trader/core version  # Should return 404 if not published
   ```

### Publishing Order

**IMPORTANT**: Packages must be published in dependency order to ensure proper resolution.

#### Phase 1: Core Package
```bash
cd packages/core
npm publish --access public
```

Wait 30 seconds for registry propagation.

#### Phase 2: MCP Protocol
```bash
cd packages/mcp-protocol
npm publish --access public
```

Wait 30 seconds for registry propagation.

#### Phase 3: Feature Packages (can be parallel)
```bash
# These packages only depend on core and can be published in any order
cd packages/backtesting && npm publish --access public
cd packages/brokers && npm publish --access public
cd packages/execution && npm publish --access public
cd packages/features && npm publish --access public
cd packages/market-data && npm publish --access public
cd packages/neural && npm publish --access public
cd packages/news-trading && npm publish --access public
cd packages/portfolio && npm publish --access public
cd packages/prediction-markets && npm publish --access public
cd packages/risk && npm publish --access public
cd packages/sports-betting && npm publish --access public
cd packages/strategies && npm publish --access public
```

Wait 30 seconds for registry propagation.

#### Phase 4: MCP Server
```bash
cd packages/mcp
npm publish --access public
```

Wait 30 seconds for registry propagation.

#### Phase 5: Meta Package
```bash
cd packages/neural-trader
npm publish --access public
```

### Automated Publishing

Use the provided script for automated publishing:

```bash
cd /workspaces/neural-trader/neural-trader-rust/packages
bash scripts/publish-all.sh
```

The script handles:
- Dependency order
- Build steps
- Error handling
- Verification with retries
- Logging
- Claude-Flow hook integration

## Verification

### Manual Verification

Check each package on npm:
```bash
npm view @neural-trader/core
npm view @neural-trader/mcp
npm view neural-trader
```

### Automated Verification

```bash
for pkg in "@neural-trader/core" "@neural-trader/mcp" "neural-trader"; do
  echo -n "$pkg: "
  npm view "$pkg" version 2>/dev/null || echo "NOT FOUND"
done
```

## Common Issues

### 1. Package Already Published

**Error**: `403 Forbidden - You cannot publish over the previously published versions`

**Solution**: Increment version in package.json:
```bash
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0
```

### 2. Authentication Required

**Error**: `npm ERR! code ENEEDAUTH`

**Solution**:
```bash
npm login
# Follow prompts to authenticate
```

### 3. Verification Timeout

**Error**: Package published but `npm view` returns 404

**Solution**: Wait 1-2 minutes for registry propagation. This is normal.

### 4. Repository URL Warning

**Warning**: `repository.url was normalized`

**Solution**: This is cosmetic. Update package.json:
```json
{
  "repository": {
    "type": "git",
    "url": "git+https://github.com/ruvnet/neural-trader.git"
  }
}
```

## Package Maintenance

### Updating Packages

1. **Make changes** to source code
2. **Update version** in package.json
3. **Build** if necessary
4. **Test** locally with `npm pack`
5. **Publish** with `npm publish --access public`

### Versioning Strategy

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Deprecating Packages

```bash
npm deprecate @neural-trader/core@1.0.0 "Use version 1.1.0 or higher"
```

### Unpublishing (CAUTION)

Only possible within 72 hours:
```bash
npm unpublish @neural-trader/core@1.0.0
```

**WARNING**: Unpublishing is discouraged. Use deprecation instead.

## Security Best Practices

1. **Never commit secrets** to package.json or source code
2. **Use npm audit** regularly: `npm audit`
3. **Enable 2FA** on npm account
4. **Review dependencies** for vulnerabilities
5. **Use scoped packages** (@neural-trader/*) for organization control

## Monitoring

### Download Statistics

Check package downloads:
```bash
npm view neural-trader downloads
```

Or visit: https://www.npmjs.com/package/neural-trader

### Package Health

Monitor on npm:
- https://www.npmjs.com/package/neural-trader
- Check for reported issues
- Review dependent packages

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Publish to NPM

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'
      - run: npm ci
      - run: npm run build
      - run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

## Support

For issues with publishing:
- npm support: https://www.npmjs.com/support
- GitHub issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: See PUBLISH_LOG.md for detailed logs

## References

- [npm Publishing Guide](https://docs.npmjs.com/packages-and-modules/contributing-packages-to-the-registry)
- [Semantic Versioning](https://semver.org/)
- [npm Documentation](https://docs.npmjs.com/)
- [Scoped Packages](https://docs.npmjs.com/cli/v8/using-npm/scope)

---

Last Updated: 2025-11-13
Version: 1.0.0
