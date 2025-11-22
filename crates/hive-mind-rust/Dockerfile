# Production-Grade Multi-stage Build for Hive Mind Rust Financial Trading System
# Optimized for 99.99% uptime requirements and financial-grade security

ARG RUST_VERSION=1.75
ARG DEBIAN_VERSION=bullseye

# Build stage with security hardening
FROM rust:${RUST_VERSION}-${DEBIAN_VERSION} as builder

# Security: Update system and install minimal required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    pkg-config=0.29.2-1 \
    libssl-dev=1.1.1n-0+deb11u5 \
    libprotobuf-dev=3.12.4-1 \
    protobuf-compiler=3.12.4-1 \
    cmake=3.18.4-2+deb11u1 \
    clang=1:11.0-51+nmu5 \
    libpq-dev=13.14-0+deb11u1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security: Create non-root build user
RUN useradd -r -s /bin/false -m -d /build builduser

# Set working directory and ownership
WORKDIR /build
RUN chown builduser:builduser /build

# Switch to build user for security
USER builduser

# Copy dependency manifests first for better layer caching
COPY --chown=builduser:builduser Cargo.toml Cargo.lock ./

# Create dummy source structure for dependency caching
RUN mkdir src && echo "fn main() {}" > src/main.rs \
    && mkdir src/bin && echo "fn main() {}" > src/bin/main.rs

# Build dependencies with production optimizations
ENV CARGO_TARGET_DIR=/build/target
RUN cargo build --release --locked \
    && rm -rf src/main.rs src/bin/main.rs

# Copy actual source code
COPY --chown=builduser:builduser src/ ./src/
COPY --chown=builduser:builduser config/ ./config/

# Build the application with maximum optimizations
RUN cargo build --release --locked --bin hive-mind-server \
    && strip target/release/hive-mind-server

# Security scan stage (commented for CI/CD integration)
# FROM aquasec/trivy:latest as security-scan
# COPY --from=builder /build/target/release/hive-mind-server /tmp/hive-mind-server
# RUN trivy fs --exit-code 1 --severity HIGH,CRITICAL /tmp/

# Production runtime stage with minimal attack surface
FROM debian:${DEBIAN_VERSION}-slim as runtime

# Security labels
LABEL security.scan="trivy" \
      security.compliance="financial-grade" \
      security.hardened="true" \
      maintainer="Ximera Production Team" \
      version="1.0.0" \
      description="Hive Mind Rust Financial Trading System"

# Install only essential runtime dependencies with specific versions
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates=20210119 \
    curl=7.74.0-1.3+deb11u11 \
    libssl1.1=1.1.1n-0+deb11u5 \
    libpq5=13.14-0+deb11u1 \
    tini=0.19.0-1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security: Create dedicated application user with minimal privileges
RUN groupadd -r -g 1000 hive-mind \
    && useradd -r -u 1000 -g hive-mind -s /bin/false \
       -c "Hive Mind Application User" \
       --no-create-home hive-mind

# Create application directory structure
RUN mkdir -p /app/{bin,config,data,logs,certs} \
    && chown -R hive-mind:hive-mind /app

# Set secure working directory
WORKDIR /app

# Copy binary with proper ownership
COPY --from=builder --chown=hive-mind:hive-mind \
    /build/target/release/hive-mind-server /app/bin/hive-mind-server

# Copy configuration with proper ownership
COPY --chown=hive-mind:hive-mind config/ /app/config/

# Security: Set proper file permissions
RUN chmod 755 /app/bin/hive-mind-server \
    && chmod -R 644 /app/config/ \
    && chmod 700 /app/data /app/logs

# Switch to non-root user
USER hive-mind

# Health check for Kubernetes readiness/liveness probes
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8091/health || exit 1

# Expose required ports
EXPOSE 8080/tcp 8090/tcp 9090/tcp 8091/tcp

# Environment variables with security considerations
ENV RUST_LOG=info \
    RUST_BACKTRACE=1 \
    HIVE_MIND_CONFIG=/app/config/production.toml \
    HIVE_MIND_DATA_DIR=/app/data \
    HIVE_MIND_LOG_DIR=/app/logs \
    MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start the application with proper signal handling
CMD ["/app/bin/hive-mind-server", "start", "--config", "/app/config/production.toml", "--daemon"]

# Security: Run container filesystem as read-only
# Mount volumes for: /app/data, /app/logs, /tmp
# Network: Use host networking or custom bridge for low latency
# Resources: Set memory/CPU limits in Kubernetes