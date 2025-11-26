# Multi-stage Dockerfile for Neural Trader Rust
# Optimized for minimal size and security

# Stage 1: Build
FROM rust:1.75-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    musl-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

# Build dependencies (cached layer)
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --workspace
RUN rm -rf src

# Copy source code
COPY src/ src/
COPY benches/ benches/
COPY examples/ examples/
COPY tests/ tests/

# Build the actual application
RUN cargo build --release --workspace

# Run tests to ensure build is valid
RUN cargo test --release --workspace

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash neural && \
    mkdir -p /app /var/lib/neural-trader /var/log/neural-trader && \
    chown -R neural:neural /app /var/lib/neural-trader /var/log/neural-trader

WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/target/release/neural-trader /usr/local/bin/neural-trader
COPY --from=builder /app/target/release/neural-trader-cli /usr/local/bin/neural-trader-cli

# Copy configuration templates
COPY .config/ /app/.config/

# Set user
USER neural

# Environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV NEURAL_TRADER_CONFIG=/app/config.toml

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD neural-trader health || exit 1

# Default command
CMD ["neural-trader", "--config", "/app/config.toml", "run"]
