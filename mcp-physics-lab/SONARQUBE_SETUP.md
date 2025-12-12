# SonarQube MCP Server Setup

Integrate SonarQube code quality analysis with Claude Code via MCP.

## Prerequisites

You need **one** of these:
1. **Docker** (recommended) - `brew install --cask docker`
2. **Java 21+** - `brew install openjdk@21`

## Option 1: Docker (Recommended)

### Install Docker
```bash
brew install --cask docker
open -a Docker  # Start Docker Desktop
```

### Start SonarQube Community Edition
```bash
# Pull and run SonarQube
docker run -d --name sonarqube \
  -p 9000:9000 \
  sonarqube:community

# Wait for startup (check http://localhost:9000)
# Default login: admin/admin
```

### Generate API Token
1. Open http://localhost:9000
2. Login (default: admin/admin, change on first login)
3. Go to **My Account** → **Security** → **Generate Tokens**
4. Create a token named `mcp-server` with type **User Token**
5. Copy the token

### Configure MCP
Add to your MCP settings (Windsurf/Claude Desktop):

```json
{
  "mcpServers": {
    "sonarqube": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "SONARQUBE_TOKEN",
        "-e", "SONARQUBE_URL",
        "mcp/sonarqube"
      ],
      "env": {
        "SONARQUBE_TOKEN": "YOUR_TOKEN_HERE",
        "SONARQUBE_URL": "http://host.docker.internal:9000"
      }
    }
  }
}
```

> **Note**: Use `host.docker.internal` instead of `localhost` when running MCP in Docker.

## Option 2: Native Java Installation

### Install Java 21
```bash
brew install openjdk@21
export JAVA_HOME=/usr/local/opt/openjdk@21
export PATH="$JAVA_HOME/bin:$PATH"
```

### Download SonarQube MCP Server
```bash
# Clone and build
git clone https://github.com/SonarSource/sonarqube-mcp-server.git
cd sonarqube-mcp-server
./gradlew clean build -x test

# JAR will be in build/libs/
```

### Configure MCP (Java)
```json
{
  "mcpServers": {
    "sonarqube": {
      "command": "java",
      "args": [
        "-jar",
        "/path/to/sonarqube-mcp-server/build/libs/sonarqube-mcp-server.jar"
      ],
      "env": {
        "SONARQUBE_TOKEN": "YOUR_TOKEN_HERE",
        "SONARQUBE_URL": "http://localhost:9000",
        "STORAGE_PATH": "/tmp/sonarqube-mcp"
      }
    }
  }
}
```

## Windsurf Plugin (Easiest)

1. Click **Plugins** button in Cascade view
2. Search for `sonarqube`
3. Click **Install**
4. Configure:
   - Token: Your SonarQube user token
   - URL: `http://localhost:9000`

## Available Tools

Once configured, Claude can use these SonarQube tools:

| Tool Category | Tools |
|---------------|-------|
| **Analysis** | `analyze_code` - Analyze code snippets |
| **Issues** | `get_issues`, `search_issues`, `get_issue_details` |
| **Projects** | `list_projects`, `get_project`, `create_project` |
| **Quality Gates** | `get_quality_gate_status`, `list_quality_gates` |
| **Rules** | `search_rules`, `get_rule_details` |
| **Measures** | `get_measures`, `get_metrics` |
| **Sources** | `get_source_code`, `get_scm_info` |

## Example Usage

Once configured, ask Claude:

```
"Analyze this code for security issues using SonarQube"

"What are the open bugs in project 'openworm'?"

"Show me the code smells in file src/main.rs"

"What's the quality gate status for my project?"
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SONARQUBE_TOKEN` | User token from SonarQube | ✅ |
| `SONARQUBE_URL` | SonarQube server URL | ✅ (for Server) |
| `SONARQUBE_ORG` | Organization key | ✅ (for Cloud) |
| `SONARQUBE_TOOLSETS` | Enabled toolsets (comma-separated) | ❌ |
| `SONARQUBE_READ_ONLY` | Disable write operations | ❌ |
| `STORAGE_PATH` | Local storage path | ❌ (Java only) |

## Toolset Configuration

Enable specific toolsets to reduce context overhead:

```json
{
  "env": {
    "SONARQUBE_TOOLSETS": "analysis,issues,quality-gates"
  }
}
```

Available toolsets:
- `analysis` - Code analysis
- `issues` - Issue management
- `projects` - Project management
- `quality-gates` - Quality gate status
- `rules` - Rule information
- `sources` - Source code access
- `measures` - Metrics and measures
- `languages` - Supported languages
- `system` - System information
- `webhooks` - Webhook management
- `dependency-risks` - Dependency analysis

## Troubleshooting

### SonarQube not accessible
```bash
# Check if SonarQube is running
curl http://localhost:9000/api/system/status

# Check Docker container
docker ps | grep sonarqube
docker logs sonarqube
```

### Token issues
- Ensure token is of type **User Token** (not Project or Global)
- Check token hasn't expired
- Verify permissions

### Docker networking
When MCP runs in Docker, use `host.docker.internal:9000` instead of `localhost:9000`

## Complete MCP Configuration

Full configuration with all servers:

```json
{
  "mcpServers": {
    "physics-lab": {
      "command": "/Volumes/Tengritek/Ashina/.venv/bin/python",
      "args": ["/Volumes/Tengritek/Ashina/HyperPhysics/mcp-physics-lab/physics_mcp_server.py"]
    },
    "sonarqube": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "SONARQUBE_TOKEN", "-e", "SONARQUBE_URL", "mcp/sonarqube"],
      "env": {
        "SONARQUBE_TOKEN": "YOUR_TOKEN",
        "SONARQUBE_URL": "http://host.docker.internal:9000"
      }
    },
    "wolfram": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-wolfram"],
      "env": {
        "WOLFRAM_APP_ID": "YOUR_APP_ID"
      }
    }
  }
}
```
