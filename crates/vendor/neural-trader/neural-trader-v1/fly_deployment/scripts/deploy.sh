#!/bin/bash

# Fly.io GPU Deployment Script for ruvtrade
# This script handles the complete deployment process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="ruvtrade"
FLYCTL_PATH="/home/codespace/.fly/bin/flyctl"
DEPLOYMENT_DIR="/workspaces/ai-news-trader/fly_deployment"

echo -e "${BLUE}🚀 Starting Fly.io GPU deployment for ${APP_NAME}${NC}"

# Check if flyctl is available
if [ ! -f "$FLYCTL_PATH" ]; then
    echo -e "${RED}❌ flyctl not found at $FLYCTL_PATH${NC}"
    echo -e "${YELLOW}Please install flyctl first${NC}"
    exit 1
fi

# Check if user is authenticated
echo -e "${BLUE}🔐 Checking authentication...${NC}"
if ! $FLYCTL_PATH auth whoami > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Not authenticated with Fly.io${NC}"
    echo -e "${BLUE}Please run: $FLYCTL_PATH auth login${NC}"
    exit 1
fi

# Change to deployment directory
cd "$DEPLOYMENT_DIR"

# Create app if it doesn't exist
echo -e "${BLUE}📱 Checking if app exists...${NC}"
if ! $FLYCTL_PATH apps list | grep -q "$APP_NAME"; then
    echo -e "${YELLOW}Creating new app: $APP_NAME${NC}"
    $FLYCTL_PATH apps create "$APP_NAME"
else
    echo -e "${GREEN}✅ App $APP_NAME already exists${NC}"
fi

# Set regions
echo -e "${BLUE}🌍 Configuring regions...${NC}"
$FLYCTL_PATH regions set ord fra nrt syd --app "$APP_NAME"

# Create volume if it doesn't exist
echo -e "${BLUE}💾 Setting up persistent storage...${NC}"
if ! $FLYCTL_PATH volumes list --app "$APP_NAME" | grep -q "ruvtrade_data"; then
    echo -e "${YELLOW}Creating volume: ruvtrade_data${NC}"
    $FLYCTL_PATH volumes create ruvtrade_data --region ord --size 10 --app "$APP_NAME"
else
    echo -e "${GREEN}✅ Volume ruvtrade_data already exists${NC}"
fi

# Set secrets (prompt for sensitive data)
echo -e "${BLUE}🔒 Setting up secrets...${NC}"
echo -e "${YELLOW}Please provide the following secrets:${NC}"

read -p "API Key for trading data: " -s API_KEY
echo
read -p "Redis URL (or press enter for default): " REDIS_URL
read -p "Database URL (or press enter for default): " DATABASE_URL

# Set default values if empty
REDIS_URL=${REDIS_URL:-"redis://localhost:6379"}
DATABASE_URL=${DATABASE_URL:-"sqlite:///app/data/trading.db"}

echo -e "${BLUE}Setting secrets...${NC}"
$FLYCTL_PATH secrets set \
    API_KEY="$API_KEY" \
    REDIS_URL="$REDIS_URL" \
    DATABASE_URL="$DATABASE_URL" \
    --app "$APP_NAME"

# Build and deploy
echo -e "${BLUE}🏗️ Building and deploying application...${NC}"
$FLYCTL_PATH deploy --app "$APP_NAME" --config fly.toml

# Check deployment status
echo -e "${BLUE}🔍 Checking deployment status...${NC}"
$FLYCTL_PATH status --app "$APP_NAME"

# Show app info
echo -e "${BLUE}📊 Application information:${NC}"
$FLYCTL_PATH info --app "$APP_NAME"

# Show logs
echo -e "${BLUE}📜 Recent logs:${NC}"
$FLYCTL_PATH logs --app "$APP_NAME" --lines 20

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${GREEN}Your app is available at: https://$APP_NAME.fly.dev${NC}"
echo -e "${BLUE}Monitor with: $FLYCTL_PATH logs --app $APP_NAME${NC}"
echo -e "${BLUE}Scale with: $FLYCTL_PATH scale --app $APP_NAME${NC}"

# Optional: Open app in browser
read -p "Open app in browser? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    $FLYCTL_PATH open --app "$APP_NAME"
fi