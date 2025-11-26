# Deployment Guide

## Overview

Comprehensive deployment guide for `@neural-trader/backend` across various platforms and environments.

## Platform Support Matrix

### Operating Systems

| OS | x64 | ARM64 | ARM | RISC-V | Status |
|---|---|---|---|---|---|
| **Linux (glibc)** | ✅ | ✅ | ✅ | ✅ | Production Ready |
| **Linux (musl/Alpine)** | ✅ | ✅ | ✅ | ✅ | Production Ready |
| **macOS (Intel)** | ✅ | N/A | N/A | N/A | Production Ready |
| **macOS (Apple Silicon)** | N/A | ✅ | N/A | N/A | Production Ready |
| **Windows** | ✅ | 🚧 | N/A | N/A | Production Ready (x64) |
| **FreeBSD** | ✅ | N/A | N/A | N/A | Beta |
| **Android** | N/A | 🚧 | 🚧 | N/A | Experimental |

### Node.js Versions

| Version | Status | Notes |
|---|---|---|
| **v14.x** | ✅ Supported | Minimum version |
| **v16.x** | ✅ Supported | LTS |
| **v18.x** | ✅ Recommended | Active LTS |
| **v20.x** | ✅ Recommended | Active LTS |
| **v22.x** | ✅ Supported | Current |

### NAPI Versions

- **NAPI v6+**: Required
- **NAPI v8**: Recommended for optimal performance
- **NAPI v9**: Supported on Node.js 18+

## Docker Deployment

### Standard Dockerfile

```dockerfile
FROM node:20-alpine

# Install build dependencies for native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    libc6-compat

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy application code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Change ownership
RUN chown -R nodejs:nodejs /app

# Switch to non-root user
USER nodejs

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => r.statusCode === 200 ? process.exit(0) : process.exit(1))"

CMD ["node", "server.js"]
```

### Multi-stage Build

```dockerfile
# Builder stage
FROM node:20-alpine AS builder

RUN apk add --no-cache python3 make g++

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production stage
FROM node:20-alpine

RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy only necessary files
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./

RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

RUN chown -R nodejs:nodejs /app

USER nodejs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD node -e "require('http').get('http://localhost:3000/health', (r) => r.statusCode === 200 ? process.exit(0) : process.exit(1))"

CMD ["node", "dist/main.js"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  neural-trader-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - JWT_SECRET=${JWT_SECRET}
      - PORT=3000
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    networks:
      - neural-trader-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - neural-trader-network

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=neuraltrader
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=trading
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - neural-trader-network

networks:
  neural-trader-network:
    driver: bridge

volumes:
  redis-data:
  postgres-data:
```

## Kubernetes Deployment

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-trader-api
  namespace: trading
  labels:
    app: neural-trader
    component: api
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: neural-trader
      component: api
  template:
    metadata:
      labels:
        app: neural-trader
        component: api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "3000"
        prometheus.io/path: "/metrics"
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - neural-trader
              topologyKey: kubernetes.io/hostname
      containers:
      - name: api
        image: neural-trader-api:2.1.1
        imagePullPolicy: Always
        ports:
        - containerPort: 3000
          name: http
          protocol: TCP
        env:
        - name: NODE_ENV
          value: "production"
        - name: PORT
          value: "3000"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: neural-trader-secrets
              key: jwt-secret
        - name: DB_HOST
          value: postgres-service
        - name: DB_PORT
          value: "5432"
        - name: DB_USER
          valueFrom:
            secretKeyRef:
              name: neural-trader-secrets
              key: db-user
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neural-trader-secrets
              key: db-password
        - name: REDIS_HOST
          value: redis-service
        - name: REDIS_PORT
          value: "6379"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2000m"
          requests:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 3000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        securityContext:
          runAsNonRoot: true
          runAsUser: 1001
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: data
          mountPath: /app/data
      volumes:
      - name: tmp
        emptyDir: {}
      - name: data
        persistentVolumeClaim:
          claimName: neural-trader-data
      securityContext:
        fsGroup: 1001
```

### Service Configuration

```yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-trader-service
  namespace: trading
  labels:
    app: neural-trader
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 3000
    protocol: TCP
    name: http
  selector:
    app: neural-trader
    component: api
---
apiVersion: v1
kind: Service
metadata:
  name: neural-trader-lb
  namespace: trading
spec:
  type: LoadBalancer
  ports:
  - port: 443
    targetPort: 3000
    protocol: TCP
    name: https
  selector:
    app: neural-trader
    component: api
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-trader-hpa
  namespace: trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-trader-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
```

## Serverless Deployment

### AWS Lambda

```javascript
// lambda-handler.js
const serverless = require('serverless-http');
const express = require('express');
const backend = require('@neural-trader/backend');

const app = express();

// Initialize backend on cold start
let initialized = false;

async function ensureInitialized() {
  if (!initialized) {
    await backend.initNeuralTrader(JSON.stringify({
      logLevel: 'info',
      enableGpu: false
    }));

    backend.initAuth(process.env.JWT_SECRET);
    backend.initRateLimiter({
      maxRequestsPerMinute: 100,
      burstSize: 20,
      windowDurationSecs: 60
    });

    initialized = true;
  }
}

app.use(express.json());

app.get('/health', async (req, res) => {
  await ensureInitialized();
  const health = await backend.healthCheck();
  res.json(health);
});

app.get('/api/analysis/:symbol', async (req, res) => {
  await ensureInitialized();

  try {
    const result = await backend.quickAnalysis(req.params.symbol);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports.handler = serverless(app);
```

### SAM Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    Runtime: nodejs20.x
    MemorySize: 2048
    Environment:
      Variables:
        NODE_ENV: production
        JWT_SECRET: !Ref JWTSecret

Resources:
  NeuralTraderFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./
      Handler: lambda-handler.handler
      Events:
        ApiRoot:
          Type: Api
          Properties:
            Path: /
            Method: ANY
        ApiGreedy:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
      Policies:
        - AWSLambdaBasicExecutionRole
      VpcConfig:
        SecurityGroupIds:
          - !Ref LambdaSecurityGroup
        SubnetIds:
          - !Ref PrivateSubnet1
          - !Ref PrivateSubnet2

  JWTSecret:
    Type: AWS::SecretsManager::Secret
    Properties:
      Description: JWT secret for neural-trader
      GenerateSecretString:
        SecretStringTemplate: '{}'
        GenerateStringKey: secret
        PasswordLength: 64

Outputs:
  ApiUrl:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/"
```

### Google Cloud Functions

```javascript
// index.js
const functions = require('@google-cloud/functions-framework');
const backend = require('@neural-trader/backend');

let initialized = false;

async function ensureInitialized() {
  if (!initialized) {
    await backend.initNeuralTrader(JSON.stringify({
      logLevel: 'info',
      enableGpu: false
    }));

    backend.initAuth(process.env.JWT_SECRET);
    initialized = true;
  }
}

functions.http('neuralTraderApi', async (req, res) => {
  await ensureInitialized();

  res.set('Access-Control-Allow-Origin', '*');

  if (req.path === '/health') {
    const health = await backend.healthCheck();
    res.json(health);
    return;
  }

  if (req.path.startsWith('/api/analysis/')) {
    const symbol = req.path.split('/').pop();
    try {
      const result = await backend.quickAnalysis(symbol);
      res.json(result);
    } catch (error) {
      res.status(500).json({ error: error.message });
    }
    return;
  }

  res.status(404).json({ error: 'Not found' });
});
```

## Edge Computing

### Cloudflare Workers

**Note**: Native modules are not supported in Cloudflare Workers. Use the full Node.js deployment for neural-trader.

### Deno Deploy

**Note**: Native modules require Node.js compatibility layer. Recommended to use traditional Node.js deployment.

## E2B Sandbox Deployment

### Sandbox Creation

```javascript
const backend = require('@neural-trader/backend');

async function deployToE2B() {
  // Create sandbox
  const sandbox = await backend.createE2bSandbox(
    'trading-agent-1',
    'nodejs'
  );

  console.log('Sandbox created:', sandbox.sandboxId);

  // Install dependencies
  await backend.executeE2bProcess(
    sandbox.sandboxId,
    'npm install @neural-trader/backend express'
  );

  // Upload server code
  const serverCode = `
    const express = require('express');
    const backend = require('@neural-trader/backend');

    const app = express();

    app.get('/health', async (req, res) => {
      const health = await backend.healthCheck();
      res.json(health);
    });

    app.listen(3000, () => {
      console.log('Server running on port 3000');
    });
  `;

  // Write server file
  await backend.executeE2bProcess(
    sandbox.sandboxId,
    `echo '${serverCode}' > server.js`
  );

  // Start server
  const result = await backend.executeE2bProcess(
    sandbox.sandboxId,
    'node server.js'
  );

  console.log('Server started:', result);
}

deployToE2B();
```

### Swarm Deployment

```javascript
async function deployTradingSwarm() {
  // Initialize swarm
  const swarm = await backend.initE2bSwarm('mesh', JSON.stringify({
    topology: 'mesh',
    maxAgents: 5,
    distributionStrategy: 'adaptive',
    enableGpu: false,
    autoScaling: true,
    minAgents: 3,
    maxMemoryMb: 2048,
    timeoutSecs: 300
  }));

  console.log('Swarm initialized:', swarm.swarmId);

  // Deploy agents
  const symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'];

  for (const symbol of symbols) {
    const agent = await backend.deployTradingAgent(
      swarm.swarmId,
      'momentum',
      [symbol],
      JSON.stringify({
        shortPeriod: 10,
        longPeriod: 20,
        threshold: 0.02
      })
    );

    console.log(`Agent deployed for ${symbol}:`, agent.agentId);
  }

  // Execute strategy
  const execution = await backend.executeSwarmStrategy(
    swarm.swarmId,
    'momentum',
    symbols
  );

  console.log('Strategy executed:', execution);

  // Monitor performance
  const metrics = await backend.getSwarmMetrics(swarm.swarmId);
  console.log('Swarm metrics:', metrics);
}

deployTradingSwarm();
```

## Platform-Specific Considerations

### Alpine Linux (musl libc)

```dockerfile
FROM node:20-alpine

# Required for musl compatibility
RUN apk add --no-cache \
    libc6-compat \
    libstdc++ \
    libgcc

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

CMD ["node", "server.js"]
```

### ARM64 (Apple Silicon, AWS Graviton)

```bash
# Cross-compile for ARM64
npm install --platform=linux --arch=arm64 @neural-trader/backend

# Or build natively on ARM64
npm install @neural-trader/backend
```

### Windows Server

```powershell
# Install Visual C++ Build Tools
npm install --global windows-build-tools

# Install neural-trader
npm install @neural-trader/backend

# Run server
node server.js
```

## Environment Variables

### Required

```bash
JWT_SECRET=your-64-byte-secret-here
```

### Recommended

```bash
NODE_ENV=production
PORT=3000
LOG_LEVEL=info
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECS=60
ENABLE_AUDIT_LOGGING=true
REQUIRE_HTTPS=true
CORS_ALLOWED_ORIGINS=https://yourdomain.com
```

### Optional

```bash
ENABLE_GPU=false
DB_HOST=localhost
DB_PORT=5432
DB_USER=neuraltrader
DB_PASSWORD=password
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Health Checks

### HTTP Endpoint

```javascript
app.get('/health', async (req, res) => {
  try {
    const health = await backend.healthCheck();

    // Check dependencies
    const checks = {
      backend: health.status === 'healthy',
      database: await checkDatabase(),
      redis: await checkRedis()
    };

    const allHealthy = Object.values(checks).every(Boolean);

    res.status(allHealthy ? 200 : 503).json({
      status: allHealthy ? 'healthy' : 'unhealthy',
      timestamp: new Date().toISOString(),
      checks
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message
    });
  }
});
```

## Monitoring and Logging

### Prometheus Metrics

```javascript
const prometheus = require('prom-client');

const register = new prometheus.Registry();

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

### Structured Logging

```javascript
const winston = require('winston');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'error.log', level: 'error' }),
    new winston.transports.File({ filename: 'combined.log' })
  ]
});
```

## Backup and Recovery

### Database Backups

```bash
#!/bin/bash
# backup.sh

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup PostgreSQL
pg_dump -h $DB_HOST -U $DB_USER trading > $BACKUP_DIR/trading_$TIMESTAMP.sql

# Compress
gzip $BACKUP_DIR/trading_$TIMESTAMP.sql

# Upload to S3
aws s3 cp $BACKUP_DIR/trading_$TIMESTAMP.sql.gz s3://neural-trader-backups/
```

## Security Hardening

### HTTPS Configuration

```javascript
const https = require('https');
const fs = require('fs');

const options = {
  key: fs.readFileSync('path/to/private-key.pem'),
  cert: fs.readFileSync('path/to/certificate.pem'),
  ca: fs.readFileSync('path/to/ca.pem')
};

https.createServer(options, app).listen(443);
```

### Secrets Management

```bash
# AWS Secrets Manager
aws secretsmanager create-secret \
  --name neural-trader/jwt-secret \
  --secret-string "your-secret-here"

# Kubernetes Secret
kubectl create secret generic neural-trader-secrets \
  --from-literal=jwt-secret=your-secret-here \
  --namespace=trading
```

## Troubleshooting

### Common Issues

1. **Native module not loading**
   - Solution: Check platform compatibility, rebuild modules

2. **High memory usage**
   - Solution: Increase Node.js heap size: `node --max-old-space-size=4096 server.js`

3. **Slow performance**
   - Solution: Enable clustering, optimize database queries, add caching

4. **Connection timeouts**
   - Solution: Increase timeout values, check network configuration

## Performance Optimization

1. **Enable clustering**: Use multiple CPU cores
2. **Add caching**: Redis for frequently accessed data
3. **Connection pooling**: Reuse database connections
4. **Load balancing**: Distribute requests across instances
5. **CDN**: Cache static assets
6. **Compression**: Enable gzip compression

## Best Practices

1. Always use HTTPS in production
2. Implement health checks for orchestration
3. Use environment variables for configuration
4. Enable monitoring and logging
5. Implement graceful shutdown
6. Use containerization for consistency
7. Automate deployments with CI/CD
8. Regular security updates
9. Backup data regularly
10. Test disaster recovery procedures
