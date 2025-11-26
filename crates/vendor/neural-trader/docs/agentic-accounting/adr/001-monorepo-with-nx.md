# ADR 001: Use Nx for Monorepo Management

**Status**: Accepted

**Date**: 2025-11-16

**Deciders**: System Architect

## Context

The agentic accounting system consists of multiple packages (core, rust-core, agents, types, mcp, api, cli) that need to be developed, tested, and deployed together. We need a monorepo tool that provides:

- Efficient task execution with caching
- TypeScript project references support
- Build orchestration with dependency management
- Good developer experience with IDE integration
- Support for mixed language projects (TypeScript + Rust)

## Decision

We will use **Nx** as our monorepo management tool instead of alternatives like Turborepo, Lerna, or yarn/pnpm workspaces alone.

## Rationale

### Nx Advantages:
1. **Computation Caching**: Nx caches build outputs and test results, providing 2-10x speedup in CI/CD
2. **Dependency Graph**: Automatic detection of package dependencies for optimal build ordering
3. **Affected Commands**: Only rebuild/test packages affected by changes (`nx affected:test`)
4. **TypeScript Support**: First-class support for TypeScript project references
5. **Extensibility**: Plugin system supports Rust builds via custom executors
6. **Distributed Execution**: Cloud-based distributed task execution (optional)
7. **Visualization**: Built-in dependency graph visualization (`nx graph`)

### Comparison with Alternatives:

| Feature | Nx | Turborepo | Lerna | Workspaces Only |
|---------|-----|-----------|-------|-----------------|
| Build Caching | ✅ Local + Remote | ✅ Local + Remote | ❌ | ❌ |
| Dependency Graph | ✅ Automatic | ✅ Manual | ✅ Manual | ❌ |
| Affected Detection | ✅ Smart | ✅ Basic | ✅ Basic | ❌ |
| Mixed Languages | ✅ Via plugins | ⚠️ Limited | ⚠️ Limited | ❌ |
| IDE Integration | ✅ Excellent | ⚠️ Good | ⚠️ Basic | ✅ Good |
| Learning Curve | ⚠️ Moderate | ✅ Low | ✅ Low | ✅ Very Low |

## Consequences

### Positive:
- **Fast CI/CD**: Computation caching reduces build times by 60-80%
- **Parallel Builds**: Nx automatically parallelizes independent tasks
- **Better DX**: IDE integration with go-to-definition across packages
- **Scalability**: Can handle 100+ packages efficiently
- **Type Safety**: TypeScript project references ensure compile-time safety

### Negative:
- **Learning Curve**: Team needs to learn Nx CLI and concepts (project.json, executors)
- **Configuration**: More setup compared to bare workspaces
- **Lock-in**: Harder to migrate away from Nx compared to simpler tools

### Mitigation:
- Provide Nx training and documentation to team
- Use Nx presets to minimize manual configuration
- Keep package.json scripts as primary interface (Nx is implementation detail)

## Implementation

1. Install Nx globally: `npm install -g nx`
2. Configure `nx.json` with caching strategies
3. Define project configurations in each package's `project.json`
4. Set up custom executors for Rust builds via napi-rs
5. Configure CI to use Nx Cloud (optional) for remote caching

## References

- [Nx Documentation](https://nx.dev)
- [Nx with Rust](https://nx.dev/recipes/other/rust)
- [TypeScript Project References](https://www.typescriptlang.org/docs/handbook/project-references.html)
