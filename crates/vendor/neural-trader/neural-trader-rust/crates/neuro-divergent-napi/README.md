# neuro-divergent-napi

NAPI bindings for the Neuro-Divergent neural forecasting library.

This crate provides Node.js bindings using NAPI-RS for the `neuro-divergent` Rust crate,
exposing 27+ neural forecasting models to JavaScript/TypeScript.

## Features

- **High Performance**: Native Rust implementation with zero-copy operations
- **Async/Await**: Full async support with Promise-based API
- **Type Safe**: Complete TypeScript definitions
- **Cross-Platform**: Linux, macOS, Windows support (x64, ARM64)

## Building

```bash
# Build for current platform
cargo build --release

# Build via NAPI
napi build --platform --release
```

## Testing

```bash
cargo test
```

## NPM Package

The NPM package is published as `@neural-trader/neuro-divergent` from the
`packages/neuro-divergent` directory.

## License

MIT OR Apache-2.0
