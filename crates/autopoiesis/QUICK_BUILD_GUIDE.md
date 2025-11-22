# Quick Build Performance Guide

## Fast Development Commands

```bash
# Fastest compilation (53.4s) - Core development
cargo check --no-default-features

# Fast compile profile (~30s) - Basic development  
cargo check --features="fast-compile"

# Full features (24.8s) - Complete system
cargo check --all-features

# Production build
cargo build --release --features="production"

# ML development only
cargo check --features="ml"

# Testing without heavy features
cargo test --no-default-features
```

## Build Profiles

| Profile | Command | Time | Use Case |
|---------|---------|------|----------|
| Minimal | `--no-default-features` | 53.4s | CI/CD, Core dev |
| Fast | `--features="fast-compile"` | ~30s | Dev iterations |
| ML | `--features="ml"` | ~35s | ML development |
| Full | `--all-features` | 24.8s | Complete system |
| Production | `--release --features="production"` | ~45s | Deployment |

## Performance Tips

1. **Use incremental compilation** - Already configured
2. **Feature-based development** - Only compile what you need
3. **Parallel builds** - Configured for maximum CPU usage
4. **Fast linker** - Using lld when available

## Troubleshooting

- **Import errors**: Check feature flags are enabled
- **Slow builds**: Use minimal features for development
- **Memory issues**: Use `cargo clean` if needed
- **CI/CD**: Use `--no-default-features` for fastest pipeline