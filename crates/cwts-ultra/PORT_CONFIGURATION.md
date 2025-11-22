# CWTS Ultra Port Configuration Guide

## Current Port Status ✅

All required ports are **AVAILABLE** for CWTS Ultra:

| Service | Port | Status | Purpose |
|---------|------|--------|---------|
| MCP WebSocket | 3000 | ✅ Available | Main trading server WebSocket endpoint |
| MCP HTTP API | 3001 | ✅ Available | REST API for control plane |
| Prometheus | 9090 | ✅ Available | Metrics and monitoring |
| Health Check | 8080 | ✅ Available | System health endpoint |

## System Ports Currently In Use

The following ports are used by other services on your system:

| Port | Service | Description |
|------|---------|-------------|
| 53 | DNS | System DNS resolver |
| 1716 | kdeconnect | KDE Connect service |
| 5355 | LLMNR | Link-Local Multicast Name Resolution |
| 8765 | python3 | Python service (possibly Jupyter or similar) |
| 36743 | language_server | Language server protocol |
| 38413 | windsurf | Windsurf editor |
| 46015 | language_server | Another language server |
| 64088 | windsurf | Windsurf editor |

## Configuration Files

### 1. Port Configuration
**Location**: `~/.local/cwts-ultra/config/ports.conf`

This file defines all port assignments and alternatives.

### 2. Environment Variables
**Location**: `~/.local/cwts-ultra/config/ports.env`

Source this file to override default ports:
```bash
source ~/.local/cwts-ultra/config/ports.env
```

### 3. Main Configuration
**Location**: `~/.local/cwts-ultra/config/production.toml`

Contains the main service configuration with port settings.

## Port Management Tools

### Check Port Availability
```bash
~/.local/cwts-ultra/scripts/check-ports.sh
```
This script:
- Checks all configured ports
- Identifies conflicts
- Suggests alternatives
- Saves overrides automatically

### Change Ports

#### Method 1: Environment Variables (Temporary)
```bash
export MCP_SERVER_PORT=5010
export MCP_HTTP_PORT=5011
~/.local/cwts-ultra/scripts/launch.sh
```

#### Method 2: Configuration File (Permanent)
Edit `~/.local/cwts-ultra/config/ports.env`:
```bash
export MCP_SERVER_PORT=5010
export MCP_HTTP_PORT=5011
export PROMETHEUS_PORT=9190
export HEALTH_CHECK_PORT=8180
```

#### Method 3: Direct in TOML (System-wide)
Edit `~/.local/cwts-ultra/config/production.toml`:
```toml
[mcp]
mcp_port = 5010
mcp_http_port = 5011

[monitoring]
metrics_port = 9190
health_check_port = 8180
```

## Alternative Port Ranges

If default ports are blocked, use these alternatives:

| Service | Default | Alternative 1 | Alternative 2 | Range |
|---------|---------|--------------|--------------|-------|
| MCP WebSocket | 3000 | 5010 | 10000 | 10000-10100 |
| MCP HTTP | 3001 | 5011 | 10001 | 10000-10100 |
| Prometheus | 9090 | 9190 | 10090 | 10000-10100 |
| Health | 8080 | 8180 | 10080 | 10000-10100 |

## Running Multiple Instances

To run multiple CWTS Ultra instances:

```bash
# Instance 1
export CWTS_INSTANCE_ID=1
export MCP_SERVER_PORT=3001
export MCP_HTTP_PORT=3002
~/.local/cwts-ultra/bin/cwts-ultra --config instance1.toml

# Instance 2
export CWTS_INSTANCE_ID=2
export MCP_SERVER_PORT=3003
export MCP_HTTP_PORT=3004
~/.local/cwts-ultra/bin/cwts-ultra --config instance2.toml
```

## Security Considerations

### Local-Only Binding (Default - Secure)
```bash
export MCP_BIND_ADDR=127.0.0.1  # Local only
```

### External Access (Use with Caution)
```bash
export MCP_BIND_ADDR=0.0.0.0    # Listen on all interfaces
```

**WARNING**: External access should only be enabled:
- Behind a firewall
- With TLS/SSL enabled
- With authentication configured
- On trusted networks only

## Firewall Configuration

If you need external access, configure firewall rules:

```bash
# UFW (Ubuntu/Debian)
sudo ufw allow from 192.168.1.0/24 to any port 3000 comment 'CWTS MCP'
sudo ufw allow from 192.168.1.0/24 to any port 9090 comment 'CWTS Metrics'

# firewalld (RHEL/Fedora)
sudo firewall-cmd --add-port=3000/tcp --permanent
sudo firewall-cmd --add-port=9090/tcp --permanent
sudo firewall-cmd --reload

# iptables (Generic)
sudo iptables -A INPUT -p tcp --dport 3000 -s 192.168.1.0/24 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 9090 -s 192.168.1.0/24 -j ACCEPT
```

## Troubleshooting

### Port Already in Use
```bash
# Find what's using a port
lsof -i :3000
sudo netstat -tlnp | grep 3000

# Kill process using port (with caution)
kill $(lsof -t -i:3000)
```

### Cannot Bind to Port
- Check permissions (ports < 1024 require root)
- Verify firewall isn't blocking
- Ensure no other instance is running
- Check SELinux/AppArmor policies

### Connection Refused
- Verify service is running
- Check bind address (127.0.0.1 vs 0.0.0.0)
- Confirm firewall rules
- Test with telnet or nc:
  ```bash
  telnet localhost 3000
  nc -zv localhost 3000
  ```

## Quick Reference

### Check All Ports
```bash
~/.local/cwts-ultra/scripts/check-ports.sh
```

### Start with Custom Ports
```bash
MCP_SERVER_PORT=5010 ~/.local/cwts-ultra/scripts/launch.sh
```

### View Current Configuration
```bash
grep -E "port|Port" ~/.local/cwts-ultra/config/production.toml
```

### Monitor Port Usage
```bash
watch -n 1 'ss -tuln | grep -E "(3000|3001|9090|8080)"'
```

## Summary

Your CWTS Ultra installation is configured with optimal ports that don't conflict with existing services. All default ports (3000, 3001, 9090, 8080) are available and ready to use.

To start the system:
```bash
~/.local/cwts-ultra/scripts/launch.sh
```

The system will automatically check port availability and warn if there are conflicts.