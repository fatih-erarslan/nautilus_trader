# Multi-stage Dockerfile for CDFA Unified
# Optimized for production deployment with multiple runtime options

# Build arguments
ARG RUST_VERSION=1.70
ARG FEATURES=full-performance
ARG PROFILE=release
ARG TARGET_ARCH=x86_64-unknown-linux-gnu

#############################################################################
# Base builder stage
#############################################################################
FROM rust:${RUST_VERSION}-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libc6-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 app

# Set working directory
WORKDIR /app

# Copy dependency manifests
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./

# Create dummy main to cache dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release && rm -rf src

# Copy source code
COPY src ./src
COPY benches ./benches
COPY examples ./examples
COPY scripts ./scripts

# Build arguments for configuration
ARG FEATURES
ARG PROFILE
ARG TARGET_ARCH

# Configure build environment
ENV RUSTFLAGS="-C target-cpu=native -C link-arg=-s"
ENV CARGO_BUILD_JOBS=4

# Build the application
RUN touch src/lib.rs && \
    cargo build --profile ${PROFILE} --features ${FEATURES} && \
    strip target/${PROFILE}/libcdfa_unified.so 2>/dev/null || true

#############################################################################
# Runtime base stage
#############################################################################
FROM debian:bookworm-slim as runtime-base

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app user
RUN useradd -m -u 1001 app

#############################################################################
# Minimal runtime stage
#############################################################################
FROM runtime-base as minimal

ARG PROFILE

# Copy built library
COPY --from=builder /app/target/${PROFILE}/libcdfa_unified.* /usr/local/lib/

# Copy configuration files
COPY --from=builder /app/scripts /usr/local/bin/scripts

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Switch to app user
USER app
WORKDIR /home/app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ldconfig -p | grep cdfa_unified || exit 1

CMD ["/bin/bash"]

#############################################################################
# Development runtime stage
#############################################################################
FROM runtime-base as development

# Install development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for development
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ARG PROFILE

# Copy everything for development
COPY --from=builder /app /app
WORKDIR /app

# Copy built artifacts
COPY --from=builder /app/target/${PROFILE} ./target/${PROFILE}

# Install cargo tools
RUN cargo install cargo-watch cargo-audit

# Switch to app user
USER app
WORKDIR /home/app

CMD ["/bin/bash"]

#############################################################################
# Production runtime stage  
#############################################################################
FROM runtime-base as production

ARG PROFILE

# Install production monitoring tools
RUN apt-get update && apt-get install -y \
    htop \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built library and binaries
COPY --from=builder /app/target/${PROFILE}/libcdfa_unified.* /usr/local/lib/
COPY --from=builder /app/scripts/performance_validation.rs /usr/local/bin/cdfa-validate
COPY --from=builder /app/examples/performance_demo /usr/local/bin/cdfa-demo

# Copy configuration and documentation
COPY --from=builder /app/docs /usr/share/doc/cdfa-unified/
COPY --from=builder /app/scripts /usr/local/share/cdfa-unified/scripts/

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Create directories for data and configuration
RUN mkdir -p /var/lib/cdfa-unified /etc/cdfa-unified && \
    chown app:app /var/lib/cdfa-unified /etc/cdfa-unified

# Copy example configuration
COPY examples/config.toml /etc/cdfa-unified/config.toml.example

# Switch to app user
USER app
WORKDIR /home/app

# Health check with actual library test
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/local/bin/cdfa-validate || exit 1

# Default command
CMD ["/usr/local/bin/cdfa-demo"]

#############################################################################
# Python integration stage
#############################################################################
FROM python:3.11-slim as python-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    matplotlib \
    jupyter

ARG PROFILE

# Copy Python bindings
COPY --from=builder /app/target/${PROFILE}/libcdfa_unified.so /usr/local/lib/python3.11/site-packages/

# Copy Python integration files
COPY python/ /usr/local/lib/python3.11/site-packages/cdfa_unified/
COPY examples/python/ /home/app/examples/

# Create app user
RUN useradd -m -u 1001 app
USER app
WORKDIR /home/app

# Set Python path
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages:$PYTHONPATH

CMD ["python3"]

#############################################################################
# Distributed computing stage
#############################################################################
FROM runtime-base as distributed

# Install distributed computing dependencies
RUN apt-get update && apt-get install -y \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

ARG PROFILE

# Copy distributed binaries and libraries
COPY --from=builder /app/target/${PROFILE}/libcdfa_unified.* /usr/local/lib/
COPY --from=builder /app/scripts /usr/local/share/cdfa-unified/scripts/

# Copy distributed configuration
COPY examples/distributed/ /etc/cdfa-unified/distributed/

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Redis configuration
COPY docker/redis.conf /etc/redis/redis.conf

# Create app user
RUN useradd -m -u 1001 app
USER app
WORKDIR /home/app

# Expose Redis port
EXPOSE 6379

# Health check for distributed services
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD redis-cli ping && /usr/local/bin/cdfa-validate || exit 1

# Start script for distributed services
COPY --chown=app:app docker/start-distributed.sh /home/app/start.sh
RUN chmod +x /home/app/start.sh

CMD ["/home/app/start.sh"]

#############################################################################
# Multi-arch build support
#############################################################################
FROM --platform=$TARGETPLATFORM runtime-base as multi-arch

ARG TARGETPLATFORM
ARG PROFILE

# Copy platform-specific binaries
COPY --from=builder /app/target/${PROFILE}/libcdfa_unified.* /usr/local/lib/

# Platform-specific optimizations
RUN case "$TARGETPLATFORM" in \
    "linux/amd64") echo "x86_64 optimizations" ;; \
    "linux/arm64") echo "ARM64 optimizations" ;; \
    "linux/arm/v7") echo "ARMv7 optimizations" ;; \
    *) echo "Generic optimizations" ;; \
    esac

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Switch to app user
USER app
WORKDIR /home/app

CMD ["/bin/bash"]

#############################################################################
# Default stage selection
#############################################################################
FROM production as final

# Labels for metadata
LABEL maintainer="CDFA Development Team <swarm@tengri.ai>"
LABEL version="0.1.0"
LABEL description="CDFA Unified - Cross-Domain Feature Alignment Library"
LABEL org.opencontainers.image.title="CDFA Unified"
LABEL org.opencontainers.image.description="High-performance cross-domain feature alignment library"
LABEL org.opencontainers.image.url="https://github.com/tengri/nautilus-trader"
LABEL org.opencontainers.image.documentation="https://docs.cdfa-unified.org"
LABEL org.opencontainers.image.source="https://github.com/tengri/nautilus-trader/crates/cdfa-unified"
LABEL org.opencontainers.image.vendor="TENGRI Trading"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"

# Build arguments as labels
ARG FEATURES
ARG PROFILE
ARG TARGET_ARCH
ARG RUST_VERSION

LABEL build.features=${FEATURES}
LABEL build.profile=${PROFILE}
LABEL build.target=${TARGET_ARCH}
LABEL build.rust-version=${RUST_VERSION}
LABEL build.timestamp=${BUILD_TIMESTAMP}

# Environment variables for runtime configuration
ENV CDFA_LOG_LEVEL=info
ENV CDFA_CONFIG_PATH=/etc/cdfa-unified/config.toml
ENV CDFA_DATA_PATH=/var/lib/cdfa-unified

# Volume for persistent data
VOLUME ["/var/lib/cdfa-unified"]

# Default ports for distributed mode
EXPOSE 6379 8080

# Final metadata
LABEL org.opencontainers.image.created=${BUILD_TIMESTAMP}