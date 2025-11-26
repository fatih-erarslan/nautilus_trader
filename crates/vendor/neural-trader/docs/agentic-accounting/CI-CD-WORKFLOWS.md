# Agentic Accounting - CI/CD Workflows

## Overview

This document describes the GitHub Actions CI/CD workflows configured for the Agentic Accounting system.

## Workflow Files

### 1. Test Suite (`agentic-accounting-test.yml`)

**Trigger**: Push to main/develop/claude/** branches, PRs to main/develop

**Jobs**:
- **Unit Tests (TypeScript)**: Runs Jest unit tests for all TypeScript packages
- **Unit Tests (Rust)**: Runs cargo test across all platforms (Linux, macOS, Windows)
- **Integration Tests**: Tests with PostgreSQL (pgvector) and Redis services
- **E2E Tests**: Full workflow end-to-end testing
- **Performance Tests**: Rust benchmark suite
- **Test Summary**: Aggregates all test results

**Coverage**: Uploads coverage to Codecov with 90% threshold

**Estimated Runtime**: 15-30 minutes

**Status Check**: Required for merging PRs

---

### 2. Code Quality & Linting (`agentic-accounting-lint.yml`)

**Trigger**: Push to main/develop/claude/** branches, PRs to main/develop

**Jobs**:
- **ESLint**: TypeScript/JavaScript linting with max-warnings=0
- **Prettier**: Format checking for all source files
- **TypeScript**: Type checking with `tsc --noEmit`
- **Rust Format**: `cargo fmt` formatting verification
- **Rust Clippy**: Comprehensive Rust linting with warnings-as-errors
- **Security Audit**: npm audit and cargo audit for vulnerabilities
- **Code Complexity**: Complexity analysis for maintainability
- **Quality Gate**: Aggregates all results

**Estimated Runtime**: 10-15 minutes

**Status Check**: Required for merging PRs

---

### 3. Build & Compile (`agentic-accounting-build.yml`)

**Trigger**: Push to main/develop/claude/** branches, PRs to main/develop

**Jobs**:
- **Build TypeScript**: Compiles all TS packages (core, agents, mcp-server, api, cli)
- **Build Rust Addon**: Multi-platform napi-rs compilation
  - Linux (x64, ARM64)
  - macOS (x64, ARM64)
  - Windows (x64)
- **Build Rust Core**: Workspace compilation
- **Validate Packages**: Checks package.json integrity
- **Build Documentation**: TypeDoc + cargo doc
- **Build Summary**: Results aggregation

**Artifacts**: All compiled binaries uploaded for 7 days

**Estimated Runtime**: 20-30 minutes

**Status Check**: Required for merging PRs

---

### 4. Code Coverage (`agentic-accounting-coverage.yml`)

**Trigger**:
- Push to main/develop
- PRs to main/develop
- Weekly schedule (Sunday midnight)

**Jobs**:
- **Coverage**: Generates comprehensive coverage reports
  - TypeScript: Jest with LCOV/HTML reports
  - Rust: cargo-tarpaulin with XML/HTML reports
- **Coverage Trend**: Tracks coverage history over time
- **Coverage Gate**: Enforces 90% minimum coverage

**Integrations**:
- Codecov for visualization
- PR comments with coverage diff
- Coverage badges generation

**Estimated Runtime**: 25-30 minutes

---

### 5. Security Scanning (`agentic-accounting-security.yml`)

**Trigger**:
- Push to main/develop
- PRs to main/develop
- Daily schedule (midnight)

**Jobs**:
- **Dependency Scan**: npm audit + cargo audit
- **CodeQL Analysis**: GitHub advanced security scanning
- **Secret Scan**: Gitleaks for credential detection
- **SAST**: Semgrep static analysis (OWASP Top 10, security rules)
- **Container Security**: Trivy vulnerability scanning
- **License Check**: Validates open-source licenses
- **SBOM**: Generates Software Bill of Materials
- **Security Summary**: Aggregates all security findings

**Estimated Runtime**: 15-20 minutes

**Status Check**: Advisory (warnings don't block PRs)

---

## Workflow Architecture

```
┌─────────────────────────────────────────────────────┐
│         Pull Request / Push to Branch               │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬─────────────┐
    │            │            │             │
    ▼            ▼            ▼             ▼
┌────────┐  ┌────────┐  ┌────────┐   ┌──────────┐
│ Lint   │  │ Test   │  │ Build  │   │ Security │
└────┬───┘  └───┬────┘  └───┬────┘   └────┬─────┘
     │          │            │             │
     └──────────┴────────────┴─────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │  Quality Gate    │
            │  (All must pass) │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Merge Allowed   │
            └──────────────────┘
```

---

## Caching Strategy

All workflows implement aggressive caching for faster builds:

### Node.js Dependencies
```yaml
- uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
```

### Cargo Registry & Build
```yaml
- uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/registry
      ~/.cargo/git
      neural-trader-rust/target
    key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
```

**Cache Hit Rate**: ~85% on typical PRs
**Time Saved**: 3-5 minutes per workflow run

---

## Service Dependencies

### PostgreSQL (pgvector)
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    env:
      POSTGRES_PASSWORD: postgres
      POSTGRES_USER: postgres
      POSTGRES_DB: agentic_accounting_test
    ports:
      - 5432:5432
```

### Redis
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - 6379:6379
```

---

## Matrix Builds

### Rust Multi-Platform
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    rust: [stable]
```

### Rust Addon Targets
```yaml
strategy:
  matrix:
    include:
      - target: x86_64-unknown-linux-gnu
      - target: aarch64-unknown-linux-gnu
      - target: x86_64-apple-darwin
      - target: aarch64-apple-darwin
      - target: x86_64-pc-windows-msvc
```

---

## Status Badges

Add these badges to your README.md:

```markdown
[![Test Suite](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-test.yml/badge.svg)](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-test.yml)
[![Code Quality](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-lint.yml/badge.svg)](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-lint.yml)
[![Build](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-build.yml/badge.svg)](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-build.yml)
[![Coverage](https://codecov.io/gh/ruvnet/neural-trader/branch/main/graph/badge.svg)](https://codecov.io/gh/ruvnet/neural-trader)
[![Security](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-security.yml/badge.svg)](https://github.com/ruvnet/neural-trader/actions/workflows/agentic-accounting-security.yml)
```

---

## Environment Variables

### Required Secrets
- `CODECOV_TOKEN`: For coverage uploads
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Optional Secrets
- `NPM_TOKEN`: For publishing packages (future)
- `CARGO_REGISTRY_TOKEN`: For publishing Rust crates (future)

---

## Performance Benchmarks

| Workflow | Cold Start | Cached Run | Parallel Jobs |
|----------|------------|------------|---------------|
| Test Suite | ~30 min | ~18 min | 5 |
| Lint | ~15 min | ~8 min | 7 |
| Build | ~35 min | ~22 min | 8 |
| Coverage | ~30 min | ~20 min | 3 |
| Security | ~20 min | ~12 min | 7 |

**Total CI Time (all workflows)**: ~25-30 minutes with caching

---

## Branch Protection Rules

Recommended settings for `main` and `develop` branches:

### Required Status Checks
- ✅ Test Suite / test-summary
- ✅ Code Quality / quality-gate
- ✅ Build / build-summary

### Additional Settings
- [x] Require branches to be up to date before merging
- [x] Require linear history
- [x] Include administrators
- [x] Allow force pushes: ❌
- [x] Allow deletions: ❌

---

## Troubleshooting

### Common Issues

**1. Cache Corruption**
```bash
# Clear cache via GitHub UI or:
git commit --allow-empty -m "chore: clear CI cache"
git push
```

**2. PostgreSQL Connection Failures**
```yaml
# Ensure health checks are configured:
options: >-
  --health-cmd pg_isready
  --health-interval 10s
  --health-timeout 5s
  --health-retries 5
```

**3. Rust Compilation Timeouts**
```yaml
# Increase timeout:
timeout-minutes: 30
```

**4. Test Flakiness**
```yaml
# Reduce parallelism:
--maxWorkers=2
```

---

## Local Testing

Run workflows locally with [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Run specific workflow
act -W .github/workflows/agentic-accounting-test.yml

# Run with secrets
act -s CODECOV_TOKEN=xxx -W .github/workflows/agentic-accounting-test.yml
```

---

## Maintenance

### Weekly Tasks
- [ ] Review security scan results
- [ ] Check coverage trends
- [ ] Update dependencies if needed

### Monthly Tasks
- [ ] Review workflow performance metrics
- [ ] Optimize cache strategies
- [ ] Update action versions

### Quarterly Tasks
- [ ] Audit all GitHub Actions for security updates
- [ ] Review and adjust resource allocations
- [ ] Benchmark against industry standards

---

## Future Enhancements

- [ ] Automated changelog generation
- [ ] Release automation workflow
- [ ] Deployment to staging/production
- [ ] Performance regression detection
- [ ] Visual regression testing
- [ ] Automated dependency updates (Dependabot)
- [ ] Mutation testing for test quality

---

## Support

For issues with CI/CD workflows:
1. Check workflow logs in GitHub Actions tab
2. Review this documentation
3. Create an issue in the repository
4. Contact the DevOps team

---

**Last Updated**: 2025-11-16
**Maintained By**: DevOps Team
**Version**: 1.0.0
