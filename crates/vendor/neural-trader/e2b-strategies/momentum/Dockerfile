# Multi-stage Dockerfile for production momentum strategy
# Optimized for size and security

# Builder stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev for build)
RUN npm ci

# Copy source code
COPY *.js ./

# Production stage
FROM node:18-alpine

# Install dumb-init for proper signal handling
RUN apk add --no-cache dumb-init curl

# Create non-root user
RUN addgroup -g 1001 trader && \
    adduser -D -u 1001 -G trader trader

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install production dependencies only
RUN npm ci --only=production && \
    npm cache clean --force

# Copy application from builder
COPY --from=builder /app/*.js ./

# Set ownership
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start application
CMD ["node", "index-optimized.js"]

# Metadata
LABEL maintainer="Neural Trader Team"
LABEL description="Production-optimized momentum trading strategy"
LABEL version="2.0.0"
