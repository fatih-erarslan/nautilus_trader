# 🧠 Hive-Mind / Swarm Intelligence API Guide

## Overview

The Swarm API provides intelligent multi-agent coordination using Claude Flow's non-interactive modes. This enables your FastAPI application to leverage distributed AI intelligence for complex problem-solving.

## Architecture

```
┌─────────────────────────────────────────────┐
│           FastAPI Application               │
├─────────────────────────────────────────────┤
│              Swarm API Layer                │
├─────────────────────────────────────────────┤
│         Claude Flow Orchestration           │
├─────────────────────────────────────────────┤
│     Multi-Agent Swarm Intelligence          │
└─────────────────────────────────────────────┘
```

## Available Endpoints

### 🐝 Core Swarm Endpoints

#### 1. Deploy Swarm
```http
POST /swarm/deploy
```

Deploy an intelligent multi-agent swarm for complex objectives.

**Request Body:**
```json
{
  "objective": "Optimize database performance",
  "strategy": "optimization",
  "mode": "distributed",
  "max_agents": 5,
  "parallel": true,
  "background": true,
  "analysis_only": false
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "status": "started",
  "message": "Swarm deployed in background",
  "monitor_url": "/swarm/status/{session_id}"
}
```

#### 2. Deploy Hive-Mind
```http
POST /swarm/hive-mind
```

Deploy a queen-led hive-mind system for complex coordination.

**Request Body:**
```json
{
  "objective": "Build microservices architecture",
  "queen_type": "adaptive",
  "max_workers": 8,
  "consensus": "majority",
  "auto_scale": true,
  "monitor": true
}
```

#### 3. Optimize Database
```http
POST /swarm/optimize-database
```

Specialized endpoint for database optimization.

**Response:**
```json
{
  "session_id": "uuid",
  "status": "started",
  "message": "Database optimization swarm deployed",
  "monitor_url": "/swarm/status/{session_id}"
}
```

#### 4. Analyze Codebase (Read-Only)
```http
POST /swarm/analyze-codebase
```

Safe analysis without code modifications.

**Response:**
```json
{
  "status": "completed",
  "analysis": {
    "security_issues": [],
    "performance_bottlenecks": [],
    "recommendations": []
  }
}
```

### 📊 Monitoring Endpoints

#### Get Swarm Status
```http
GET /swarm/status/{session_id}
```

#### Get Hive-Mind Status
```http
GET /swarm/hive-mind/status/{session_id}
```

#### List All Sessions
```http
GET /swarm/sessions
```

### 🔬 Specialized Swarms

#### Research Topic
```http
POST /swarm/research?topic=quantum-computing
```

#### Develop Feature
```http
POST /swarm/develop?feature=authentication-system
```

#### SPARC Development
```http
POST /swarm/sparc
```

Use SPARC methodology with swarm coordination.

**Request Body:**
```json
{
  "task": "Build REST API",
  "mode": "tdd"
}
```

## Usage Examples

### Python Client Example

```python
import requests
import time

# Deploy a swarm for optimization
response = requests.post(
    "http://localhost:8081/swarm/deploy",
    json={
        "objective": "Optimize API response times",
        "strategy": "optimization",
        "max_agents": 3,
        "parallel": True,
        "background": True
    }
)

session = response.json()
session_id = session["session_id"]

# Monitor progress
while True:
    status = requests.get(f"http://localhost:8081/swarm/status/{session_id}")
    result = status.json()
    
    if result["status"] == "completed":
        print("Optimization complete!")
        print(result["result"])
        break
    elif result["status"] == "failed":
        print("Optimization failed:", result.get("error"))
        break
    
    time.sleep(5)  # Check every 5 seconds
```

### JavaScript/TypeScript Example

```typescript
async function deploySwarm(objective: string) {
  const response = await fetch('http://localhost:8081/swarm/deploy', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      objective,
      strategy: 'development',
      max_agents: 5,
      parallel: true
    })
  });
  
  const result = await response.json();
  return result.session_id;
}

async function monitorSwarm(sessionId: string) {
  const response = await fetch(`/swarm/status/${sessionId}`);
  return await response.json();
}
```

### cURL Examples

```bash
# Deploy a research swarm
curl -X POST http://localhost:8081/swarm/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "AI trading strategies"}'

# Deploy hive-mind for complex task
curl -X POST http://localhost:8081/swarm/hive-mind \
  -H "Content-Type: application/json" \
  -d '{
    "objective": "Design high-frequency trading system",
    "queen_type": "strategic",
    "max_workers": 10,
    "consensus": "weighted"
  }'

# Safe codebase analysis (read-only)
curl -X POST http://localhost:8081/swarm/analyze-codebase
```

## Swarm Strategies

### Research
- Web searches
- Data gathering
- Pattern analysis
- Literature review

### Development
- Code generation
- Feature implementation
- API creation
- Testing

### Analysis
- Performance profiling
- Security auditing
- Code review
- Bottleneck detection

### Optimization
- Query optimization
- Algorithm improvement
- Resource utilization
- Speed enhancement

### Testing
- Unit tests
- Integration tests
- Load testing
- Security testing

### Maintenance
- Bug fixes
- Refactoring
- Documentation
- Updates

## Swarm Modes

### Centralized
- Single coordinator
- Top-down control
- Best for simple tasks

### Distributed
- Peer-to-peer coordination
- No single point of failure
- Best for parallel tasks

### Hierarchical
- Queen/worker structure
- Layered decision making
- Best for complex projects

### Mesh
- Full interconnection
- Collective intelligence
- Best for research

### Hybrid
- Adaptive topology
- Dynamic reorganization
- Best for unknown problems

## Best Practices

### 1. Choose the Right Strategy
```python
# For code analysis
strategy = "analysis"

# For building features
strategy = "development"

# For performance improvements
strategy = "optimization"
```

### 2. Use Background Mode for Long Tasks
```python
# Long-running tasks
background = True

# Quick tasks
background = False
```

### 3. Enable Parallel Execution
```python
# Speed up with parallel agents
parallel = True
max_agents = 5
```

### 4. Use Analysis Mode for Safety
```python
# Read-only analysis
analysis_only = True
```

## Integration with Trading System

### Optimize Trading Strategies
```python
response = requests.post(
    "/swarm/deploy",
    json={
        "objective": "Optimize momentum trading parameters",
        "strategy": "optimization",
        "max_agents": 4,
        "parallel": True
    }
)
```

### Research Market Patterns
```python
response = requests.post(
    "/swarm/research",
    params={"topic": "cryptocurrency arbitrage opportunities"}
)
```

### Develop Trading Features
```python
response = requests.post(
    "/swarm/develop",
    params={"feature": "risk management system"}
)
```

## Performance Considerations

### Resource Usage
- Each agent consumes ~50-100MB RAM
- CPU usage scales with agent count
- Network bandwidth for coordination

### Optimization Tips
1. Limit agents for simple tasks (2-3)
2. Use more agents for complex tasks (5-8)
3. Enable caching for repeated operations
4. Use analysis mode for exploration

## Error Handling

```python
try:
    response = requests.post("/swarm/deploy", json=config)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 500:
        print("Swarm deployment failed")
    elif e.response.status_code == 404:
        print("Session not found")
```

## Security Considerations

1. **Authentication**: Protect swarm endpoints with JWT
2. **Rate Limiting**: Prevent resource exhaustion
3. **Sandboxing**: Run swarms in isolated environments
4. **Audit Logging**: Track all swarm activities
5. **Resource Limits**: Cap maximum agents and memory

## Monitoring & Metrics

### Health Check
```http
GET /swarm/health
```

Returns:
```json
{
  "status": "healthy",
  "active_swarms": 2,
  "active_hive_minds": 1,
  "total_sessions": 15
}
```

## Advanced Features

### Neural Swarm Training
Train neural networks using swarm intelligence:
```python
response = requests.post(
    "/swarm/neural-swarm",
    json={
        "data_path": "/data/trading_data.csv",
        "pattern_type": "coordination"
    }
)
```

### SPARC Methodology
Use systematic development approach:
```python
response = requests.post(
    "/swarm/sparc",
    json={
        "task": "Build authentication system",
        "mode": "tdd"  # Test-driven development
    }
)
```

## Troubleshooting

### Common Issues

1. **Swarm Timeout**
   - Increase max execution time
   - Reduce agent count
   - Simplify objective

2. **Memory Errors**
   - Reduce max_agents
   - Enable auto-scaling
   - Use distributed mode

3. **Coordination Failures**
   - Check network connectivity
   - Verify Claude Flow installation
   - Review error logs

## Next Steps

1. **Enable in Production**
   - Add authentication
   - Implement rate limiting
   - Set up monitoring

2. **Customize Agents**
   - Create specialized agent types
   - Define custom strategies
   - Build domain-specific swarms

3. **Scale Operations**
   - Deploy on multiple servers
   - Use message queues
   - Implement caching

## Support

- Claude Flow Docs: https://github.com/ruvnet/claude-flow
- API Documentation: http://localhost:8081/docs#/Swarm%20Intelligence
- GitHub Issues: https://github.com/ruvnet/ai-news-trader/issues