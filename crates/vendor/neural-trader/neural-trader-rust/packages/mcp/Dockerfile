FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy application
COPY . .

# Skip build - no build step needed

# Expose port (if using HTTP transport)
EXPOSE 3000

# Run MCP server
CMD ["node", "bin/neural-trader.js"]
