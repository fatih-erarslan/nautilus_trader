#!/bin/bash
# Build script for CWTS Ultra

echo "Building CWTS Ultra Trading System..."

# Native build with maximum optimization
echo "Building native binary..."
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat -C codegen-units=1" \
cargo build --release --features simd

# WASM build (if wasm target is installed)
if rustup target list --installed | grep -q wasm32-unknown-unknown; then
    echo "Building WASM module..."
    cargo build --target wasm32-unknown-unknown --release --features simd
    
    # Optimize WASM if wasm-opt is available
    if command -v wasm-opt &> /dev/null; then
        echo "Optimizing WASM..."
        wasm-opt -O4 --enable-simd target/wasm32-unknown-unknown/release/cwts_ultra.wasm -o cwts_ultra_opt.wasm
    fi
fi

echo "Build complete!"
echo "Run with: ./target/release/cwts-ultra"