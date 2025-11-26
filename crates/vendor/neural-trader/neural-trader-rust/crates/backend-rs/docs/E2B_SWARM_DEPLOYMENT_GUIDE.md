# E2B Swarm Deployment Guide

**Version**: 1.0.0
**Date**: 2025-11-14
**Status**: Production Ready

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Dashboard Access](#dashboard-access)
4. [Deploying Agents](#deploying-agents)
5. [Monitoring Agents](#monitoring-agents)
6. [Managing Agents](#managing-agents)
7. [API Integration](#api-integration)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

The E2B Swarm Management Dashboard provides a comprehensive interface for deploying, monitoring, and managing AI agent swarms running in E2B cloud sandboxes. This guide covers all aspects of using the dashboard and integrating with the Rust backend API.

### Key Features

✅ **Real-time Monitoring** - 5-second auto-refresh for live agent status
✅ **Agent Deployment** - Simple modal-based deployment workflow
✅ **Metrics Tracking** - Token usage, cost, and performance metrics
✅ **Lifecycle Management** - Deploy, monitor, terminate, and destroy agents
✅ **Multi-template Support** - Base, Node.js, Python, React templates

---

## Prerequisites

### Backend Requirements

1. **E2B API Key** - Obtain from [e2b.dev](https://e2b.dev)
2. **Rust Backend Running** - BeClever API server on port 8001
3. **Environment Variables** - Configure `.env` file

```bash
# .env configuration
E2B_API_KEY=e2b_your_api_key_here
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here
```

### Frontend Requirements

1. **Next.js Application** - Running on development or production mode
2. **Dependencies Installed** - `npm install` or `pnpm install`
3. **Build Successful** - Verify with `npm run build`

---

## Dashboard Access

### Navigation

1. **Login** to BeClever.ai dashboard
2. **Navigate** to E2B Swarms from sidebar (Core section)
3. **Dashboard URL**: `http://localhost:3000/dashboard/swarms`

### Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  E2B Swarm Management        [Deploy Agent Button]  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  📊 Metrics Cards (4 cards in grid)                │
│  - Total Sandboxes  - Active Agents                │
│  - Tokens Used      - Total Cost                    │
│                                                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  🤖 Active Agents Table                            │
│  - Agent Name | Status | Sandbox ID                │
│  - Task | Tokens | Time | Cost | Actions           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## Deploying Agents

### Step-by-Step Deployment

#### 1. Click "Deploy Agent" Button

Located in the top-right corner of the dashboard.

#### 2. Fill Deployment Form

**Required Fields**:
- **Agent Name**: Unique identifier (e.g., `api-researcher-01`)
- **Agent Type**: Select from dropdown
  - `researcher` - Research and analysis tasks
  - `coder` - Code implementation tasks
  - `analyst` - Data analysis tasks
  - `tester` - Testing and QA tasks
- **Task Description**: Detailed task instructions

**Template Configuration**:
- **E2B Template**: Select sandbox environment
  - `base` - Minimal Linux environment
  - `nodejs` - Node.js runtime with npm
  - `python` - Python runtime with pip
  - `react` - React development environment

**Advanced Options**:
- **Environment Variables**: Key-value pairs for sandbox
- **Capabilities**: Array of agent capabilities
- **Timeout**: Execution timeout in seconds (default: 3600)

#### 3. Example Deployment

```javascript
Agent Name: "api-endpoint-analyzer"
Agent Type: "researcher"
Task: "Analyze REST API endpoints for authentication patterns"
Template: "nodejs"
Environment Variables:
  - API_URL: "https://api.example.com"
  - AUTH_TOKEN: "your_token_here"
Capabilities: ["api_analysis", "security_audit"]
Timeout: 3600
```

#### 4. Submit Deployment

Click **"Deploy Agent"** button. The modal will close and the agent will appear in the table with status "pending".

### Backend API Call

```bash
curl -X POST http://localhost:8001/api/agents/deploy \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "api-endpoint-analyzer",
    "agent_type": "researcher",
    "task_description": "Analyze REST API endpoints",
    "template": "nodejs",
    "environment": {
      "API_URL": "https://api.example.com"
    },
    "capabilities": ["api_analysis", "security_audit"],
    "config": {
      "timeout": 3600
    }
  }'
```

**Response**:
```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "sandbox_id": "i9xlhu4d0e8zp6dm64vqd",
  "status": "pending"
}
```

---

## Monitoring Agents

### Real-Time Status Updates

The dashboard automatically refreshes every **5 seconds** to show:

- ✅ **Agent Status** - pending → running → completed/failed
- 📊 **Token Usage** - Cumulative tokens consumed
- ⏱️ **Execution Time** - Milliseconds elapsed
- 💰 **Cost Tracking** - USD cost calculation
- 🆔 **Sandbox ID** - E2B sandbox identifier

### Status Indicators

| Status | Badge Color | Description |
|--------|------------|-------------|
| `pending` | 🟡 Yellow | Agent queued for deployment |
| `running` | 🔵 Blue | Agent actively executing |
| `completed` | 🟢 Green | Task finished successfully |
| `failed` | 🔴 Red | Task failed with error |

### Metrics Dashboard

**Total Sandboxes**
- Count of all deployed E2B sandboxes
- Includes active and completed agents

**Active Agents**
- Real-time count of running agents
- Excludes pending/completed/failed

**Tokens Used**
- Cumulative token consumption across all agents
- Updated on each status refresh

**Total Cost**
- USD cost calculated from token usage
- Rate: ~$0.0001 per token (example)

---

## Managing Agents

### View Agent Details

Click **"View"** button in Actions column to see:

- Complete task description
- Full sandbox configuration
- Environment variables
- Agent capabilities
- Execution logs (if available)
- Error messages (if failed)

### Terminate Agent

**Use Case**: Stop a running agent gracefully

1. Click **"Terminate"** in Actions column
2. Confirm termination in dialog
3. Agent status changes to "completed" or "failed"
4. Sandbox may remain active (configurable)

**API Call**:
```bash
DELETE http://localhost:8001/api/agents/{agent_id}?terminate_only=true
```

### Destroy Agent

**Use Case**: Complete cleanup including sandbox deletion

1. Click **"Destroy"** in Actions column
2. Confirm destruction in dialog
3. Agent is removed from database
4. E2B sandbox is deleted
5. All resources freed

**API Call**:
```bash
DELETE http://localhost:8001/api/agents/{agent_id}
```

### Bulk Operations

*Coming Soon*: Select multiple agents for bulk terminate/destroy operations.

---

## API Integration

### Backend Endpoints

#### List All Agents

```http
GET /api/agents
```

**Response**:
```json
{
  "agents": [
    {
      "id": "uuid",
      "name": "agent-name",
      "agent_type": "researcher",
      "status": "running",
      "sandbox_id": "sandbox-id",
      "task_description": "Task details",
      "tokens_used": 1500,
      "cost_usd": 0.15,
      "execution_time_ms": 45000,
      "created_at": "2025-11-14T10:00:00Z",
      "completed_at": null,
      "error_message": null
    }
  ]
}
```

#### Deploy Agent

```http
POST /api/agents/deploy
Content-Type: application/json

{
  "agent_name": "string",
  "agent_type": "researcher|coder|analyst|tester",
  "task_description": "string",
  "template": "base|nodejs|python|react",
  "environment": { "key": "value" },
  "capabilities": ["string"],
  "config": {
    "timeout": 3600
  }
}
```

#### Get Agent Status

```http
GET /api/agents/{agent_id}
```

#### Terminate/Destroy Agent

```http
DELETE /api/agents/{agent_id}?terminate_only=true
```

### Frontend Integration

```typescript
// Fetch agents
const response = await fetch('http://localhost:8001/api/agents');
const data = await response.json();
setAgents(data.agents || []);

// Deploy agent
const deployResponse = await fetch('http://localhost:8001/api/agents/deploy', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    agent_name: "my-agent",
    agent_type: "researcher",
    task_description: "Research task",
    template: "nodejs",
    environment: {},
    capabilities: [],
    config: { timeout: 3600 }
  })
});

// Terminate agent
await fetch(`http://localhost:8001/api/agents/${agentId}`, {
  method: 'DELETE'
});
```

---

## Troubleshooting

### Common Issues

#### 1. E2B API 404 Error

**Symptom**: Agent stuck in "pending" status, E2B returns 404

**Solution**: Verify E2B API configuration in `/crates/api/src/e2b_client.rs`:

```rust
// Correct configuration
base_url: "https://api.e2b.dev".to_string()  // NO /v1 suffix
.header("X-API-Key", &self.api_key)           // NOT Authorization: Bearer

#[serde(rename = "templateID")]  // NOT "template"
pub template: String,
```

#### 2. Agent Status Not Updating

**Symptom**: Dashboard shows stale data

**Possible Causes**:
- Backend API not running
- CORS issues blocking requests
- Auto-refresh disabled

**Solution**:
```bash
# Check backend is running
curl http://localhost:8001/api/agents

# Check CORS configuration in main.rs
let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods([Method::GET, Method::POST, Method::DELETE])
    .allow_headers([CONTENT_TYPE]);
```

#### 3. Deployment Fails

**Symptom**: "Failed to deploy agent" error

**Debug Steps**:
1. Check E2B API key validity
2. Verify network connectivity to E2B
3. Check backend logs for errors
4. Validate request payload format

```bash
# Test E2B API directly
curl -X POST https://api.e2b.dev/sandboxes \
  -H "X-API-Key: your_key" \
  -d '{"templateID":"base"}'
```

#### 4. High Token Usage

**Symptom**: Unexpected token consumption

**Investigation**:
1. Check agent task complexity
2. Review LLM model selection (OpenRouter)
3. Monitor execution logs
4. Optimize task descriptions

---

## Best Practices

### Agent Design

1. **Clear Task Descriptions**
   - Provide specific, actionable instructions
   - Define expected outputs
   - Set reasonable scope limits

2. **Template Selection**
   - Use minimal templates when possible
   - Match runtime to task requirements
   - Consider startup time vs capabilities

3. **Resource Management**
   - Set appropriate timeouts (default: 3600s)
   - Monitor token usage regularly
   - Clean up completed agents

### Cost Optimization

1. **Batch Similar Tasks**
   - Deploy agents for related tasks together
   - Share environment setups
   - Reuse sandboxes when possible

2. **Monitor Metrics**
   - Track cost per agent type
   - Identify expensive operations
   - Optimize frequently-run tasks

3. **Cleanup Strategy**
   - Destroy completed agents promptly
   - Set retention policies
   - Archive important results

### Security

1. **Environment Variables**
   - Never commit secrets to git
   - Use `.env` files for sensitive data
   - Rotate API keys regularly

2. **Sandbox Isolation**
   - Trust E2B's sandbox security
   - Validate agent outputs
   - Monitor for unusual behavior

3. **Access Control**
   - Implement authentication (planned)
   - Log all agent operations
   - Review deployment permissions

---

## Advanced Usage

### Custom Agent Types

Extend agent types by modifying backend enum:

```rust
// In crates/api/src/agents.rs
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AgentType {
    Researcher,
    Coder,
    Analyst,
    Tester,
    Custom(String),  // Add custom types
}
```

### Webhook Integration

*Coming Soon*: Configure webhooks for agent lifecycle events:

- `agent.deployed` - Agent starts
- `agent.completed` - Task finishes
- `agent.failed` - Task errors
- `agent.destroyed` - Cleanup complete

### Multi-Region Deployment

*Future Enhancement*: Deploy agents across multiple E2B regions for:

- Reduced latency
- Geographic distribution
- Redundancy

---

## Support & Resources

### Documentation

- E2B Documentation: https://e2b.dev/docs
- BeClever API Docs: `/dashboard/docs`
- Backend Source: `/beclever/backend-rs/crates/api`

### Related Files

- **E2B Client**: `/crates/api/src/e2b_client.rs`
- **Agent Routes**: `/crates/api/src/agents.rs`
- **Dashboard Component**: `/src/components/dashboard/SwarmDashboard.tsx`
- **Integration Status**: `/docs/FINAL_INTEGRATION_STATUS.md`

### Getting Help

- GitHub Issues: https://github.com/FoxRev/beclever/issues
- E2B Support: support@e2b.dev
- Internal Team: Contact backend team

---

**Last Updated**: 2025-11-14
**Guide Version**: 1.0.0
**Production Status**: ✅ **READY**
