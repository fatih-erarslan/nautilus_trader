# Docker Configuration

This directory contains Docker-related files for the Neural Trader project.

## Files

- **Dockerfile** - Main Docker image for production deployment
- **Dockerfile.fly** - Optimized Dockerfile for Fly.io deployment
- **docker-compose.yml** - Docker Compose configuration for local development
- **.dockerignore** - Files to exclude from Docker builds

## Usage

### Local Development with Docker Compose

```bash
# From project root
docker-compose -f docs/docker/docker-compose.yml up
```

### Build Docker Image

```bash
# From project root
docker build -f docs/docker/Dockerfile -t neural-trader .
```

### Fly.io Deployment

```bash
# From project root
flyctl deploy --dockerfile docs/docker/Dockerfile.fly
```

## Notes

- Docker files are kept in this directory to maintain a clean root directory
- All Docker commands should be run from the project root
- Environment variables should be configured in `.env` file (not tracked in git)
