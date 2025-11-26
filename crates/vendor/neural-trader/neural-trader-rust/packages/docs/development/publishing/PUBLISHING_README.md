# Neural Trader - NPM Publishing Documentation

## Overview

This directory contains the complete documentation and automation for publishing Neural Trader packages to npm. All 16 packages have been successfully published and are available at https://www.npmjs.com/package/neural-trader

## Quick Links

- **Main Package**: [neural-trader](https://www.npmjs.com/package/neural-trader)
- **Core Package**: [@neural-trader/core](https://www.npmjs.com/package/@neural-trader/core)
- **MCP Server**: [@neural-trader/mcp](https://www.npmjs.com/package/@neural-trader/mcp)
- **Neural Networks**: [@neural-trader/neural](https://www.npmjs.com/package/@neural-trader/neural)
- **Risk Management**: [@neural-trader/risk](https://www.npmjs.com/package/@neural-trader/risk)

## Installation

```bash
# Install complete suite
npm install neural-trader

# Or install specific packages
npm install @neural-trader/core
npm install @neural-trader/mcp
npm install @neural-trader/neural
```

## Documentation Files

### 1. PUBLISH_LOG.md
**Purpose**: Complete chronological log of the initial publishing process

Contains:
- Publishing timeline for all 16 packages
- Success/error messages
- Verification results
- Final status summary

**Use when**: You need to review the history of the initial v1.0.0 release

### 2. docs/NPM_PUBLISHING_GUIDE.md
**Purpose**: Comprehensive guide for future publishing operations

Contains:
- Package architecture and dependencies
- Step-by-step publishing instructions
- Common issues and solutions
- Security best practices
- CI/CD integration examples

**Use when**: You need to publish a new version or troubleshoot publishing issues

### 3. docs/PUBLISHING_SUCCESS_REPORT.md
**Purpose**: Executive summary and metrics of the publishing operation

Contains:
- Success metrics and statistics
- Published package details
- Technical implementation details
- Next steps and recommendations

**Use when**: You need high-level overview or metrics for reporting

### 4. scripts/publish-all.sh
**Purpose**: Automated publishing script

Features:
- Dependency-ordered publishing
- Automatic verification with retries
- Error handling and logging
- Claude-Flow hooks integration

**Use when**: You need to publish all packages automatically

```bash
cd /workspaces/neural-trader/neural-trader-rust/packages
bash scripts/publish-all.sh
```

### 5. .github/workflows/publish-npm.yml
**Purpose**: GitHub Actions workflow for automated CI/CD publishing

Features:
- Triggered by releases or manual workflow dispatch
- Automatic version bumping
- Build and publish automation
- GitHub release creation

**Use when**: You want to automate future releases via GitHub Actions

## Package Structure

```
neural-trader/
├── @neural-trader/core (foundation)
├── @neural-trader/mcp-protocol (protocol definitions)
├── @neural-trader/mcp (MCP server)
├── Feature Packages:
│   ├── @neural-trader/backtesting
│   ├── @neural-trader/brokers
│   ├── @neural-trader/execution
│   ├── @neural-trader/features
│   ├── @neural-trader/market-data
│   ├── @neural-trader/neural
│   ├── @neural-trader/news-trading
│   ├── @neural-trader/portfolio
│   ├── @neural-trader/prediction-markets
│   ├── @neural-trader/risk
│   ├── @neural-trader/sports-betting
│   └── @neural-trader/strategies
└── neural-trader (meta package)
```

## Publishing Workflow

### Automated (Recommended)

1. **Using GitHub Actions**:
   - Go to Actions tab in GitHub
   - Select "Publish to NPM" workflow
   - Click "Run workflow"
   - Choose version bump type (patch/minor/major)

2. **Using the script**:
   ```bash
   cd packages
   bash scripts/publish-all.sh
   ```

### Manual

1. Update version numbers in all package.json files
2. Build packages: `npm run build`
3. Publish in dependency order:
   - Core → MCP Protocol → Features → MCP Server → Meta Package

See `docs/NPM_PUBLISHING_GUIDE.md` for detailed manual instructions.

## Version Management

We follow [Semantic Versioning](https://semver.org/):

- **PATCH** (1.0.x): Bug fixes
- **MINOR** (1.x.0): New features (backward compatible)
- **MAJOR** (x.0.0): Breaking changes

## Current Status (v1.0.0)

| Package | Version | Status | Size |
|---------|---------|--------|------|
| neural-trader | 1.0.0 | ✅ Live | 2.1 KB |
| @neural-trader/core | 1.0.0 | ✅ Live | 3.4 KB |
| @neural-trader/mcp | 1.0.0 | ✅ Live | 2.5 KB |
| @neural-trader/neural | 1.0.0 | ✅ Live | 711.5 KB |
| @neural-trader/risk | 1.0.0 | ✅ Live | 711.6 KB |
| ... and 11 more packages | 1.0.0 | ✅ Live | - |

**Total**: 16 packages, ~11.5 MB

## Common Tasks

### Check Package Status
```bash
npm view neural-trader
npm view @neural-trader/core
```

### Update Package Version
```bash
cd packages/core
npm version patch  # 1.0.0 -> 1.0.1
```

### Test Package Locally
```bash
cd packages/core
npm pack
npm install -g neural-trader-core-1.0.0.tgz
```

### Verify Publishing
```bash
# Check all packages
for pkg in neural-trader @neural-trader/core @neural-trader/mcp; do
  npm view $pkg version
done
```

## Troubleshooting

### Package Not Found After Publishing
**Issue**: npm view returns 404 immediately after publishing
**Solution**: Wait 1-2 minutes for npm registry propagation

### Authentication Error
**Issue**: npm ERR! code ENEEDAUTH
**Solution**: Run `npm login` and authenticate

### Version Conflict
**Issue**: Cannot publish over existing version
**Solution**: Bump version number in package.json

See `docs/NPM_PUBLISHING_GUIDE.md` for more troubleshooting tips.

## Security

- All packages published with public access
- No secrets or API keys in source code
- 2FA recommended for npm account
- Regular security audits with `npm audit`

## Monitoring

### Download Statistics
Check on npm or use:
```bash
npm view neural-trader downloads
```

### Package Health
- Monitor on npmjs.com
- Check for security vulnerabilities
- Review dependent packages

## Support

- **Issues**: https://github.com/ruvnet/neural-trader/issues
- **Documentation**: See files listed above
- **NPM Support**: https://www.npmjs.com/support

## Contributing

When updating packages:
1. Update source code
2. Update tests
3. Update documentation
4. Bump version
5. Build and test locally
6. Publish via script or GitHub Actions

## License

See LICENSE file in repository root.

---

**Last Updated**: 2025-11-13
**Current Version**: 1.0.0
**Status**: ✅ All packages published successfully
**Publisher**: ruvnet
