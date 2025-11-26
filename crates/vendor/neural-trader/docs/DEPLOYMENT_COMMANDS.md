# Deployment Commands - E2B and Flow Nexus Integration

Complete cloud deployment system for neural-trader strategies with E2B sandbox and Flow Nexus platform integration.

## Overview

The deployment system provides comprehensive cloud deployment capabilities with:

- **E2B Sandbox**: Secure, isolated execution environments
- **Flow Nexus Platform**: Distributed swarm deployment with neural network training
- **Deployment Management**: List, monitor, scale, and control deployments
- **Real-time Monitoring**: Log streaming and status tracking
- **Template System**: Pre-configured deployment templates
- **Dry-run Mode**: Test deployments without execution

## Quick Start

### Deploy to E2B Sandbox

```bash
# Basic deployment
neural-trader deploy e2b momentum

# Production deployment with custom configuration
neural-trader deploy e2b momentum \
  --template neural-trader-optimized \
  --scale 3 \
  --cpu 4 \
  --memory 8 \
  --env-vars "NODE_ENV=production,LOG_LEVEL=info" \
  --auto-restart \
  --watch
```

### Deploy to Flow Nexus Platform

```bash
# Basic swarm deployment
neural-trader deploy flow-nexus mean-reversion --swarm 5

# Advanced deployment with neural network training
neural-trader deploy flow-nexus pairs-trading \
  --swarm 10 \
  --topology mesh \
  --neural \
  --workflow \
  --realtime \
  --auto-scale \
  --min-agents 5 \
  --max-agents 20
```

## Commands

### Deploy Commands

#### `deploy e2b <strategy>`

Deploy trading strategy to E2B sandbox.

**Options:**
- `-t, --template <name>` - E2B template (default: neural-trader-base)
- `-e, --env-vars <vars>` - Environment variables (KEY=value,KEY2=value2)
- `-s, --scale <count>` - Number of sandbox instances (default: 1)
- `-r, --region <region>` - Deployment region (default: us-east)
- `--cpu <cores>` - CPU cores per sandbox (default: 2)
- `--memory <gb>` - Memory per sandbox in GB (default: 4)
- `--timeout <seconds>` - Execution timeout (default: 3600)
- `--auto-restart` - Automatically restart on failure
- `--dry-run` - Simulate deployment without executing
- `-c, --config <file>` - Load configuration from file
- `--watch` - Watch logs after deployment

**Example:**
```bash
neural-trader deploy e2b momentum \
  --template neural-trader-optimized \
  --scale 3 \
  --cpu 4 \
  --memory 8 \
  --env-vars "API_KEY=xxx,MAX_POSITION=10000" \
  --auto-restart \
  --watch
```

#### `deploy flow-nexus <strategy>`

Deploy trading strategy to Flow Nexus platform.

**Options:**
- `-s, --swarm <count>` - Number of swarm agents (default: 1)
- `-t, --topology <type>` - Swarm topology: mesh, hierarchical, ring (default: mesh)
- `-n, --neural` - Enable neural network training
- `-e, --env-vars <vars>` - Environment variables
- `-r, --region <region>` - Deployment region (default: us-east)
- `--cpu <cores>` - CPU cores per agent (default: 2)
- `--memory <gb>` - Memory per agent in GB (default: 4)
- `--workflow` - Enable workflow automation
- `--realtime` - Enable real-time monitoring
- `--auto-scale` - Enable automatic scaling
- `--min-agents <count>` - Minimum agents for auto-scaling
- `--max-agents <count>` - Maximum agents for auto-scaling
- `--dry-run` - Simulate deployment without executing
- `-c, --config <file>` - Load configuration from file
- `--watch` - Watch deployment after creation

**Example:**
```bash
neural-trader deploy flow-nexus mean-reversion \
  --swarm 10 \
  --topology mesh \
  --neural \
  --workflow \
  --auto-scale \
  --min-agents 5 \
  --max-agents 20
```

### Management Commands

#### `deploy list`

List all deployments.

**Options:**
- `-p, --platform <platform>` - Filter by platform (e2b, flow-nexus)
- `-s, --status <status>` - Filter by status (running, stopped, failed)
- `-l, --limit <count>` - Limit number of results (default: 20)
- `--json` - Output as JSON
- `--all` - Show all deployments including deleted

**Example:**
```bash
# List all deployments
neural-trader deploy list

# List running E2B deployments
neural-trader deploy list --platform e2b --status running

# List all deployments as JSON
neural-trader deploy list --json
```

#### `deploy status <deployment-id>`

Get detailed status of a specific deployment.

**Options:**
- `--json` - Output as JSON
- `--metrics` - Include performance metrics
- `--watch` - Watch status updates
- `--refresh <seconds>` - Refresh interval for watch mode (default: 5)

**Example:**
```bash
# Get deployment status
neural-trader deploy status deploy-abc123

# Watch deployment status with metrics
neural-trader deploy status deploy-abc123 --metrics --watch --refresh 10
```

#### `deploy logs <deployment-id>`

View and stream logs from deployment.

**Options:**
- `-f, --follow` - Follow log output (stream)
- `-n, --lines <count>` - Number of lines to show (default: 100)
- `--since <time>` - Show logs since timestamp
- `--until <time>` - Show logs until timestamp
- `--filter <pattern>` - Filter logs by pattern (regex)
- `--level <level>` - Filter by log level (error, warn, info, debug)
- `--instance <id>` - Show logs from specific instance
- `--json` - Output as JSON
- `--no-color` - Disable colored output

**Example:**
```bash
# View last 100 logs
neural-trader deploy logs deploy-abc123

# Stream logs in real-time
neural-trader deploy logs deploy-abc123 --follow

# Filter error logs
neural-trader deploy logs deploy-abc123 --level error --lines 50
```

#### `deploy scale <deployment-id> <count>`

Scale deployment instances up or down.

**Options:**
- `-y, --yes` - Skip confirmation prompt
- `--strategy <strategy>` - Scaling strategy: gradual, immediate (default: gradual)
- `--max-concurrent <count>` - Maximum concurrent scaling operations (default: 3)

**Example:**
```bash
# Scale to 10 instances
neural-trader deploy scale deploy-abc123 10

# Scale immediately without confirmation
neural-trader deploy scale deploy-abc123 5 --yes --strategy immediate
```

#### `deploy stop <deployment-id>`

Stop a running deployment.

**Options:**
- `-y, --yes` - Skip confirmation prompt
- `--graceful` - Graceful shutdown with cleanup (default: true)
- `--timeout <seconds>` - Shutdown timeout (default: 30)

**Example:**
```bash
# Stop deployment gracefully
neural-trader deploy stop deploy-abc123

# Force stop without confirmation
neural-trader deploy stop deploy-abc123 --yes --timeout 10
```

#### `deploy delete <deployment-id>`

Permanently delete a deployment.

**Options:**
- `-y, --yes` - Skip confirmation prompt
- `--force` - Force delete even if running
- `--keep-data` - Keep deployment data/logs

**Example:**
```bash
# Delete deployment
neural-trader deploy delete deploy-abc123

# Force delete running deployment and keep data
neural-trader deploy delete deploy-abc123 --force --keep-data
```

## Deployment Templates

Pre-configured templates for common deployment scenarios.

### Available Templates

#### E2B Templates

**e2b-basic** - Basic single instance deployment
```json
{
  "platform": "e2b",
  "template": "neural-trader-base",
  "scale": 1,
  "resources": {
    "cpu": 2,
    "memory": 4,
    "timeout": 3600
  }
}
```

**e2b-neural-trader** - Production-ready with 3 instances
```json
{
  "platform": "e2b",
  "template": "neural-trader-optimized",
  "scale": 3,
  "autoRestart": true,
  "resources": {
    "cpu": 4,
    "memory": 8,
    "timeout": 7200
  }
}
```

#### Flow Nexus Templates

**flow-nexus-swarm** - 5-agent mesh swarm with neural network
```json
{
  "platform": "flow-nexus",
  "swarm": {
    "count": 5,
    "topology": "mesh"
  },
  "neural": true,
  "workflow": true,
  "autoScale": true
}
```

**flow-nexus-hierarchical** - Large-scale 20-agent hierarchical deployment
```json
{
  "platform": "flow-nexus",
  "swarm": {
    "count": 20,
    "topology": "hierarchical",
    "minAgents": 10,
    "maxAgents": 50
  },
  "neural": true,
  "workflow": true
}
```

### Using Templates

```bash
# Use template by name
neural-trader deploy e2b momentum --config e2b-neural-trader

# Use custom template file
neural-trader deploy e2b momentum --config ./my-config.json
```

### Template Location

Templates are stored in:
```
src/cli/templates/deploy/
  ├── e2b-basic.json
  ├── e2b-neural-trader.json
  ├── flow-nexus-swarm.json
  └── flow-nexus-hierarchical.json
```

## Architecture

### File Structure

```
src/cli/
├── commands/deploy/
│   ├── index.js           # Main deployment router
│   ├── e2b.js             # E2B deployment command
│   ├── flow-nexus.js      # Flow Nexus deployment command
│   ├── list.js            # List deployments
│   ├── status.js          # Deployment status
│   ├── logs.js            # Log streaming
│   ├── scale.js           # Scale deployment
│   ├── stop.js            # Stop deployment
│   └── delete.js          # Delete deployment
├── lib/
│   ├── e2b-manager.js     # E2B sandbox management
│   ├── flow-nexus-client.js # Flow Nexus API client
│   ├── deployment-tracker.js # Local deployment tracking
│   ├── remote-executor.js # Remote command execution
│   ├── log-streamer.js    # Real-time log streaming
│   ├── deployment-validator.js # Config validation
│   └── chalk-compat.js    # Chalk compatibility layer
└── templates/deploy/
    ├── e2b-basic.json
    ├── e2b-neural-trader.json
    ├── flow-nexus-swarm.json
    └── flow-nexus-hierarchical.json
```

### Components

#### E2B Manager (`e2b-manager.js`)

Manages E2B sandbox deployments:
- Sandbox creation and lifecycle management
- Code upload and environment configuration
- Multi-sandbox coordination
- Resource monitoring
- Log collection

#### Flow Nexus Client (`flow-nexus-client.js`)

Manages Flow Nexus platform deployments:
- Authentication and API communication
- Swarm initialization and agent spawning
- Neural network configuration
- Workflow automation setup
- Real-time monitoring

#### Deployment Tracker (`deployment-tracker.js`)

Local deployment state management:
- Persistent deployment metadata storage
- Deployment listing and filtering
- Status synchronization
- Statistics and analytics
- Import/export functionality

#### Remote Executor (`remote-executor.js`)

Execute commands on remote deployments:
- Command execution on instances
- Script upload and execution
- File transfer (upload/download)
- Batch operations across deployments

#### Log Streamer (`log-streamer.js`)

Real-time log streaming:
- Fetch historical logs
- Stream real-time updates
- Filter and search logs
- Export logs to file
- Log statistics

## Configuration

### Environment Variables

**E2B:**
```bash
E2B_API_KEY=your-api-key
```

**Flow Nexus:**
```bash
FLOW_NEXUS_TOKEN=your-token
# Or authenticate via CLI:
npx flow-nexus@latest login
```

### Deployment Configuration

Create custom deployment configurations:

```json
{
  "platform": "e2b",
  "strategy": "momentum",
  "template": "neural-trader-optimized",
  "scale": 3,
  "region": "us-east",
  "resources": {
    "cpu": 4,
    "memory": 8,
    "timeout": 7200
  },
  "autoRestart": true,
  "envVars": {
    "NODE_ENV": "production",
    "LOG_LEVEL": "info",
    "MAX_POSITION_SIZE": "10000",
    "RISK_LIMIT": "0.02"
  }
}
```

## Error Handling

All commands include comprehensive error handling:

- **Validation Errors**: Configuration validation before deployment
- **API Errors**: Clear error messages from platform APIs
- **Network Errors**: Retry logic for transient failures
- **Timeout Errors**: Configurable timeouts for long operations
- **Authentication Errors**: Clear guidance on authentication requirements

## Dry-run Mode

Test deployments without execution:

```bash
# Test E2B deployment
neural-trader deploy e2b momentum --dry-run

# Test Flow Nexus deployment
neural-trader deploy flow-nexus mean-reversion --swarm 5 --dry-run
```

Dry-run mode:
- Validates configuration
- Shows deployment summary
- Checks authentication
- Displays resource requirements
- Does NOT create actual deployments

## Monitoring and Logs

### Real-time Monitoring

```bash
# Watch deployment status
neural-trader deploy status deploy-abc123 --watch --metrics

# Stream logs
neural-trader deploy logs deploy-abc123 --follow

# Filter and search logs
neural-trader deploy logs deploy-abc123 --filter "ERROR" --level error
```

### Performance Metrics

Status command with `--metrics` shows:
- Total trades
- Success/failure rates
- Average latency
- Throughput
- Error rates
- CPU usage
- Memory usage

## Scaling

### Manual Scaling

```bash
# Scale up
neural-trader deploy scale deploy-abc123 10

# Scale down
neural-trader deploy scale deploy-abc123 3
```

### Auto-scaling (Flow Nexus)

Configure auto-scaling in deployment:

```bash
neural-trader deploy flow-nexus momentum \
  --swarm 5 \
  --auto-scale \
  --min-agents 3 \
  --max-agents 20
```

## Security

### Authentication

**E2B**: Requires API key in environment or config
**Flow Nexus**: Requires authentication via CLI or token

### Environment Variables

- Never hardcode secrets in config files
- Use environment variables for sensitive data
- Validator warns about potential secrets in env vars

### Network Security

- All API communication over HTTPS
- Sandboxed execution environments
- Isolated network access per deployment

## Troubleshooting

### Common Issues

**"E2B API key not found"**
```bash
export E2B_API_KEY=your-api-key
```

**"Not authenticated with Flow Nexus"**
```bash
npx flow-nexus@latest login
```

**"Deployment not found"**
- Check deployment ID (can use partial ID)
- Use `neural-trader deploy list` to see all deployments

**"Configuration validation failed"**
- Check error messages for specific validation failures
- Use `--dry-run` to test configuration

### Debug Mode

Enable debug logging:
```bash
neural-trader --debug deploy e2b momentum
```

## Examples

### Example 1: Basic E2B Deployment

```bash
# Deploy momentum strategy to E2B
neural-trader deploy e2b momentum

# Check status
neural-trader deploy status deploy-abc123

# View logs
neural-trader deploy logs deploy-abc123

# Stop when done
neural-trader deploy stop deploy-abc123
```

### Example 2: Production Flow Nexus Deployment

```bash
# Deploy with full configuration
neural-trader deploy flow-nexus pairs-trading \
  --swarm 10 \
  --topology hierarchical \
  --neural \
  --workflow \
  --auto-scale \
  --min-agents 5 \
  --max-agents 20 \
  --cpu 8 \
  --memory 16 \
  --env-vars "NODE_ENV=production,LOG_LEVEL=warn"

# Watch deployment
neural-trader deploy status deploy-xyz789 --watch --metrics

# Scale based on load
neural-trader deploy scale deploy-xyz789 15
```

### Example 3: Using Templates

```bash
# Use pre-configured template
neural-trader deploy e2b momentum \
  --config e2b-neural-trader \
  --env-vars "API_KEY=xxx,MAX_POSITION=10000"

# Create custom template and use it
cat > my-config.json <<EOF
{
  "platform": "e2b",
  "scale": 5,
  "autoRestart": true,
  "resources": {
    "cpu": 8,
    "memory": 16
  }
}
EOF

neural-trader deploy e2b momentum --config ./my-config.json
```

## Best Practices

1. **Use Templates**: Start with pre-configured templates for common scenarios
2. **Test with Dry-run**: Always test with `--dry-run` before actual deployment
3. **Monitor Deployments**: Use `--watch` and `--metrics` to monitor performance
4. **Auto-restart**: Enable auto-restart for production deployments
5. **Auto-scaling**: Use auto-scaling for variable workloads
6. **Log Streaming**: Stream logs during deployment for real-time monitoring
7. **Graceful Shutdown**: Always use graceful shutdown unless emergency
8. **Resource Allocation**: Allocate appropriate CPU/memory based on strategy complexity
9. **Environment Variables**: Use environment variables for configuration
10. **Regular Monitoring**: Check deployment status regularly

## Integration with MCP Tools

Deployment commands can be used alongside MCP tools:

```bash
# Initialize swarm coordination
npx claude-flow@alpha mcp-tools --action swarm_init

# Deploy with neural-trader CLI
neural-trader deploy flow-nexus momentum --swarm 5 --neural

# Monitor via MCP
npx claude-flow@alpha mcp-tools --action swarm_status
```

## Support

For issues or questions:
- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://github.com/ruvnet/neural-trader
- Flow Nexus: https://flow-nexus.ruv.io

## Version

Deployment commands version: 1.0.0
Neural Trader version: 2.3.15

---

**Note**: This is a comprehensive deployment system designed for production use. Always test deployments in a development environment before deploying to production.
