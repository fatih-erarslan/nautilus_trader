# Multi-stage build for Agentic Accounting System
FROM node:20-alpine AS builder

# Install Rust for napi-rs builds
RUN apk add --no-cache \
    curl \
    build-base \
    python3 \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY packages/agentic-accounting-core/package.json ./packages/agentic-accounting-core/
COPY packages/agentic-accounting-rust-core/package.json ./packages/agentic-accounting-rust-core/
COPY packages/agentic-accounting-agents/package.json ./packages/agentic-accounting-agents/
COPY packages/agentic-accounting-types/package.json ./packages/agentic-accounting-types/
COPY packages/agentic-accounting-api/package.json ./packages/agentic-accounting-api/
COPY packages/agentic-accounting-cli/package.json ./packages/agentic-accounting-cli/
COPY packages/agentic-accounting-mcp/package.json ./packages/agentic-accounting-mcp/

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Build Rust packages
WORKDIR /app/packages/agentic-accounting-rust-core
RUN npm run build

# Build TypeScript packages
WORKDIR /app
RUN npm run build --workspaces

# Production stage
FROM node:20-alpine

WORKDIR /app

# Copy built artifacts
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/packages ./packages
COPY --from=builder /app/package.json ./

# Create necessary directories
RUN mkdir -p /app/.agentdb /app/logs

# Install production dependencies only
RUN npm prune --production

# Expose ports
EXPOSE 3000 4000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD node -e "require('http').get('http://localhost:3000/health', (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

# Set user
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
USER nodejs

CMD ["node", "packages/agentic-accounting-api/dist/server.js"]
