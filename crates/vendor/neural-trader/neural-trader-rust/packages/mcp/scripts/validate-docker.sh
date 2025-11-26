#!/bin/bash
# Level 5: Docker Validation
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "🐳 Level 5: Docker Validation"
echo "=============================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
IMAGE_NAME="neural-trader-mcp"
CONTAINER_NAME="mcp-test-container"

cleanup() {
    echo -e "\n${YELLOW}Cleaning up Docker resources...${NC}"
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
}

trap cleanup EXIT

# 1. Check if Docker is available
echo -e "\n${YELLOW}5.1 Checking Docker availability...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓ Docker is installed${NC}"
    docker --version
else
    echo -e "${RED}✗ Docker is not installed${NC}"
    echo "Skipping Docker validation"
    exit 0
fi

# 2. Build Docker image
echo -e "\n${YELLOW}5.2 Building Docker image...${NC}"
cd "${PROJECT_ROOT}"

if [ -f "Dockerfile" ]; then
    if docker build -t $IMAGE_NAME . 2>&1 | tee /tmp/docker-build.log; then
        echo -e "${GREEN}✓ Docker image built successfully${NC}"
    else
        echo -e "${RED}✗ Docker build failed${NC}"
        ERRORS=$((ERRORS + 1))
        cat /tmp/docker-build.log
    fi
else
    echo -e "${YELLOW}⚠ No Dockerfile found${NC}"
    echo "Creating basic Dockerfile..."

    cat > Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy application
COPY . .

# Build if needed
RUN npm run build || true

# Expose port (if using HTTP transport)
EXPOSE 3000

# Run MCP server
CMD ["node", "bin/neural-trader.js"]
EOF

    docker build -t $IMAGE_NAME .
fi

# 3. Run container
echo -e "\n${YELLOW}5.3 Running container...${NC}"

if docker run -d --name $CONTAINER_NAME $IMAGE_NAME; then
    echo -e "${GREEN}✓ Container started successfully${NC}"

    # Wait for startup
    sleep 5

    # Check if container is running
    if docker ps | grep -q $CONTAINER_NAME; then
        echo -e "${GREEN}✓ Container is running${NC}"
    else
        echo -e "${RED}✗ Container stopped unexpectedly${NC}"
        docker logs $CONTAINER_NAME
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${RED}✗ Failed to start container${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 4. Test external connectivity
echo -e "\n${YELLOW}5.4 Testing external connectivity...${NC}"

if docker exec $CONTAINER_NAME node -e "console.log('Connected')" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Can execute commands in container${NC}"
else
    echo -e "${RED}✗ Cannot connect to container${NC}"
    ERRORS=$((ERRORS + 1))
fi

# 5. Check resource usage
echo -e "\n${YELLOW}5.5 Checking resource usage...${NC}"

STATS=$(docker stats $CONTAINER_NAME --no-stream --format "{{.MemUsage}}" 2>/dev/null || echo "N/A")
echo "Memory usage: $STATS"

if [ "$STATS" != "N/A" ]; then
    MEM_MB=$(echo $STATS | grep -o "[0-9.]*MiB" | grep -o "[0-9.]*" || echo "0")
    if (( $(echo "$MEM_MB < 100" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${GREEN}✓ Memory usage under 100MB baseline${NC}"
    else
        echo -e "${YELLOW}⚠ Memory usage: ${MEM_MB}MB${NC}"
    fi
fi

# 6. Multi-platform test (if supported)
echo -e "\n${YELLOW}5.6 Checking multi-platform support...${NC}"

if docker buildx version > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Docker buildx available${NC}"
    echo "Platforms supported by builder:"
    docker buildx inspect default | grep "Platforms:" || echo "  Default platforms"
else
    echo -e "${YELLOW}⚠ Docker buildx not available (multi-platform builds not supported)${NC}"
fi

# 7. Check image size
echo -e "\n${YELLOW}5.7 Checking image size...${NC}"

IMAGE_SIZE=$(docker images $IMAGE_NAME --format "{{.Size}}" | head -1)
echo "Image size: $IMAGE_SIZE"

# Parse size (rough check)
if echo "$IMAGE_SIZE" | grep -q "GB"; then
    SIZE_NUM=$(echo "$IMAGE_SIZE" | grep -o "[0-9.]*")
    if (( $(echo "$SIZE_NUM > 1" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${YELLOW}⚠ Image size is large (${IMAGE_SIZE})${NC}"
    else
        echo -e "${GREEN}✓ Image size acceptable${NC}"
    fi
else
    echo -e "${GREEN}✓ Image size acceptable (${IMAGE_SIZE})${NC}"
fi

# Summary
echo -e "\n=============================="
echo "Level 5 Summary:"
echo "  Errors: $ERRORS"
echo "  Image: $IMAGE_NAME"
echo "  Size: $IMAGE_SIZE"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Level 5: PASSED${NC}"
    exit 0
else
    echo -e "${RED}❌ Level 5: FAILED${NC}"
    exit 1
fi
