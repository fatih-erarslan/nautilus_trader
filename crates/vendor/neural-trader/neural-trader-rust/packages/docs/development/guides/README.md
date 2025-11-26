# Development Guides

Development guides, migration documentation, and improvement tracking.

## 📚 Guide Documentation

### Migration Guides
- **[MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md)** - Package migration guide
  - Migrating to modular packages
  - Breaking changes
  - Update procedures
  - Compatibility notes

### README Documentation
- **[README-TEMPLATE.md](./README-TEMPLATE.md)** - README template
  - Standard README structure
  - Required sections
  - Best practices
  - Examples

- **[README_IMPROVEMENTS.md](./README_IMPROVEMENTS.md)** - README improvements
  - Documentation enhancements
  - Clarity improvements
  - Example additions

- **[README_v1.0.5_IMPROVEMENTS.md](./README_v1.0.5_IMPROVEMENTS.md)** - Version 1.0.5 improvements
  - Version-specific updates
  - Changelog integration
  - Feature documentation

### Issue Tracking
- **[GITHUB_ISSUES_SUMMARY.md](./GITHUB_ISSUES_SUMMARY.md)** - GitHub issues summary
  - Open issues overview
  - Issue priorities (P0, P1, P2)
  - Resolution status
  - Issue links

- **[FIXES_APPLIED.md](./FIXES_APPLIED.md)** - Applied fixes documentation
  - Bug fixes
  - Improvements
  - Resolutions

## 🔄 Migration Guide

### Migrating to Modular Packages

**From monolithic package:**
```bash
# Old
npm install neural-trader

# New (minimal)
npm install @neural-trader/core @neural-trader/strategies
```

**Import changes:**
```typescript
// Old
import { Strategy } from 'neural-trader';

// New
import { Strategy } from '@neural-trader/strategies';
```

### Breaking Changes

**Version 1.0.0:**
- Modular package structure
- New import paths
- Updated dependencies
- NAPI bindings required for some packages

## 📖 README Best Practices

### Required Sections
1. **Title and Description**
2. **Installation**
3. **Quick Start**
4. **API Reference**
5. **Examples**
6. **Contributing**
7. **License**

### Optional Sections
- **Features**
- **Performance**
- **Troubleshooting**
- **FAQ**
- **Changelog**

### Code Examples
Always include:
- Installation commands
- Basic usage example
- Common use cases
- Error handling

## 🐛 Issue Tracking

### Current Issues (6 Total)

**Critical (P0) - 1 issue:**
- **#69**: Hardcoded native binding paths in execution/features packages
  - Impact: High
  - Packages: execution, features
  - Status: Open

**High (P1) - 1 issue:**
- **#70**: RSI calculation returns NaN values
  - Impact: Medium
  - Package: features
  - Status: Open

**Medium (P2) - 4 issues:**
- **#71**: Remove unnecessary dependencies from news-trading
- **#72**: Implement sports-betting and prediction-markets
- **#73**: Add test suites across packages
- **#74**: Improve documentation coverage

### Issue Resolution Process

1. **Triage**: Assign priority (P0/P1/P2)
2. **Investigate**: Reproduce and analyze
3. **Fix**: Implement solution
4. **Test**: Verify fix works
5. **Document**: Update docs
6. **Close**: Mark as resolved

## 📊 Documentation Improvements

### Version 1.0.5 Improvements
- ✅ Updated all package READMEs
- ✅ Added comprehensive examples
- ✅ Improved API documentation
- ✅ Added troubleshooting sections
- ✅ Updated installation instructions

### Planned Improvements
- [ ] Video tutorials
- [ ] Interactive examples
- [ ] API playground
- [ ] Migration tools
- [ ] Better error messages

## 🔧 Development Best Practices

### Code Style
- Use TypeScript strict mode
- Follow ESLint rules
- Write comprehensive tests
- Document all public APIs

### Documentation
- Keep READMEs up to date
- Include code examples
- Document breaking changes
- Maintain changelogs

### Testing
- Write tests first (TDD)
- Aim for 80%+ coverage
- Test edge cases
- Include integration tests

### Git Workflow
- Use conventional commits
- Create feature branches
- Write descriptive commit messages
- Reference issues in commits

## 🔗 Related Documentation

- [Testing Documentation](../testing/) - Test suite
- [Verification Documentation](../verification/) - Verification reports
- [Features Documentation](../features/) - Feature implementation

---

[← Back to Development](../README.md)
