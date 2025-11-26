# Build stage
FROM rust:1.91 AS builder

WORKDIR /app

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build in release mode
RUN cargo build --release --bin beclever-api

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/beclever-api /usr/local/bin/

# Expose port
EXPOSE 3001

# Run the application
CMD ["beclever-api"]
